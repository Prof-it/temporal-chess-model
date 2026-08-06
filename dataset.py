"""Dataset for Marvin Chess with board history and legal moves."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

__all__ = [
    "StreamingDataset",
    "create_dataloader",
    "count_samples",
]

# Policy outputs: 64*64 from-to move logits.
# Resign and flag are predicted by separate binary heads, not policy indices.
NUM_POLICY_OUTPUTS = 4096

# Time target: absolute seconds at 1s resolution. 1 bin = 1 second; the top
# class (index CAP) is the open-ended ">=CAP s" overflow. Must match model.py
# TIME_SECOND_CAP / NUM_TIME_BINS (= CAP + 1).
TIME_SECOND_CAP = 120


@dataclass
class Sample:
    """A single training sample."""
    # Board history: (8, 64) - 8 board states, each with 64 piece codes
    board_history: np.ndarray
    # Legal moves: list of (from, to, promo) tuples
    legal_moves: List[Tuple[int, int, int]]
    # Time history: (8,) - time spent on last 8 moves
    time_history: np.ndarray
    # Repetition flags: (8,) - binary flags for each history position
    repetition_flags: np.ndarray
    # Castling rights: (4,) - K, Q, k, q
    castling_rights: np.ndarray
    # EP square: -1 if none, else 0-63
    ep_square: int
    # Scalars
    active_elo: int
    opp_elo: int
    move_ply: int
    halfmove_clock: int
    # tc_cat encoding (Lichess perfType, 5 categories):
    #   0 = bullet      (T < 180 s)
    #   1 = blitz       (T < 480 s)
    #   2 = rapid       (T < 1500 s)
    #   3 = classical   (T >= 1500 s)
    #   4 = ultraBullet (T < 30 s, kept only under a small global cap)
    # Where T = base_time + 40 * inc.
    # Older shards (pre-2026-05) used 0=blitz / 1=rapid / 2=classical and lacked bullet.
    tc_cat: int
    # Initial base time of the game (seconds) — distinct from active_clock,
    # which is the remaining time at this position. Used by the time-conditioning
    # encoder so the model knows the game format (1+0 vs 90+30 etc.).
    base_time: float
    active_clock: float
    opp_clock: float
    active_inc: int
    opp_inc: int
    # True iff active_elo lies in [1200, 2500). Out-of-range rows are kept but
    # are not part of the balanced cohort and may have non-uniform coverage.
    in_balance_range: bool = True
    # Targets
    policy_move: Tuple[int, int, int] = (0, 0, 0)  # (from, to, promo)
    policy_is_resign: bool = False
    policy_is_flag: bool = False
    y_val: float = 0.0
    time_target: float = 0.0


def _sample_to_tensors(sample: Sample) -> Dict[str, torch.Tensor]:
    """Convert a Sample to model input tensors."""

    # Time history: (8,) float32 - model expects most-recent first.
    # Parquet rows from `process_pgn.py` store time_history oldest->newest.
    # Raw seconds: the time encoder applies its own log compression downstream.
    time_history = torch.from_numpy(sample.time_history[::-1].copy()).float()

    # EP square: one-hot-ish encoding - (64,) with 1 at ep_square if valid
    ep_mask = torch.zeros(64, dtype=torch.float32)
    if 0 <= sample.ep_square < 64:
        ep_mask[sample.ep_square] = 1.0

    # Scalars (layout MUST stay in sync with model.py forward() and inference/encoding.py):
    #   0 active_elo_norm   1 opp_elo_norm     2 ply_norm
    #   3 active_clock_s    4 opp_clock_s      5 active_inc_s
    #   6 opp_inc_s         7 hmc_norm         8 base_time_s
    # ELO / ply / hmc stay normalized — model code consumes them directly.
    # Time fields are RAW SECONDS — the time encoder picks its own featurization.
    active_elo_norm = (sample.active_elo - 1900) / 700.0
    opp_elo_norm = (sample.opp_elo - 1900) / 700.0
    ply_norm = sample.move_ply / 100.0
    hmc_norm = sample.halfmove_clock / 100.0

    scalars = torch.tensor([
        active_elo_norm, opp_elo_norm, ply_norm,
        float(sample.active_clock), float(sample.opp_clock),
        float(sample.active_inc), float(sample.opp_inc), hmc_norm,
        float(sample.base_time),
    ], dtype=torch.float32)

    # Legal move mask: (4096,) bool - True for legal moves
    legal_mask = torch.zeros(NUM_POLICY_OUTPUTS, dtype=torch.bool)
    for from_sq, to_sq, promo in sample.legal_moves:
        if 0 <= from_sq < 64 and 0 <= to_sq < 64:
            move_idx = from_sq * 64 + to_sq
            legal_mask[move_idx] = True

    # Policy / promo targets. Resign and flag samples have no move: policy_target
    # is the cross-entropy ignore_index sentinel (-100) and has_move=False, so
    # the policy/time/promo losses skip them. Resign/flag are trained via their
    # own binary heads (resign_target / flag_target).
    promo_target = 0
    promo_file = 0
    has_move = not (sample.policy_is_resign or sample.policy_is_flag)

    if has_move:
        from_sq, to_sq, promo = sample.policy_move
        policy_target = from_sq * 64 + to_sq
        promo_target = promo
        if 56 <= to_sq <= 63:
            promo_file = to_sq - 56
    else:
        policy_target = -100  # ignore_index for cross-entropy

    policy_target = torch.tensor(policy_target, dtype=torch.long)

    # Value target (regression: 0=loss, 0.5=draw, 1=win)
    y_val = torch.tensor(sample.y_val, dtype=torch.float32)

    # Value classification target (WDL: 0=loss, 1=draw, 2=win)
    # y_val is 0.0, 0.5, or 1.0 -> class 0, 1, 2
    y_val_cls = torch.tensor(int(sample.y_val * 2), dtype=torch.long)

    # Time target: absolute seconds at 1s resolution (matches Lichess %clk
    # granularity). 1 bin = 1 second; the top class is the open-ended ">=CAP s"
    # overflow. No dequant noise — there is nothing to dequantize at 1s bins;
    # sub-second smoothing is an inference-decode concern, not a training label.
    time_bin = min(int(round(sample.time_target)), TIME_SECOND_CAP)
    time_target_cls = torch.tensor(time_bin, dtype=torch.long)

    return {
        "board_history": torch.tensor(sample.board_history, dtype=torch.long),
        "time_history": time_history,
        "rep_flags": torch.tensor(sample.repetition_flags, dtype=torch.float32),
        "castling": torch.tensor(sample.castling_rights, dtype=torch.float32),
        "ep_mask": ep_mask,
        "scalars": scalars,
        "legal_mask": legal_mask,
        "policy_target": policy_target,
        "has_move": torch.tensor(has_move, dtype=torch.bool),
        "resign_target": torch.tensor(float(sample.policy_is_resign), dtype=torch.float32),
        "flag_target": torch.tensor(float(sample.policy_is_flag), dtype=torch.float32),
        "y_val": y_val,
        "y_val_cls": y_val_cls,
        "time_target_cls": time_target_cls,
        "promo_target": torch.tensor(promo_target, dtype=torch.long),
        "promo_file": torch.tensor(promo_file, dtype=torch.long),
        # tc_cat surfaced for per-TC metric tracking (bullet/blitz/rapid/classical/ultraBullet)
        "tc_cat": torch.tensor(sample.tc_cat, dtype=torch.long),
    }


def _sample_to_tensors_fast(
    board_history: np.ndarray,
    legal_moves: List[Tuple[int, int, int]],
    time_history: np.ndarray,
    rep_flags: np.ndarray,
    castling: np.ndarray,
    ep_square: int,
    active_elo: int,
    opp_elo: int,
    move_ply: int,
    halfmove_clock: int,
    tc_cat: int,
    base_time: float,
    active_clock: float,
    opp_clock: float,
    active_inc: int,
    opp_inc: int,
    policy_move: Tuple[int, int, int],
    policy_is_resign: bool,
    policy_is_flag: bool,
    y_val: float,
    time_target: float,
) -> Dict[str, torch.Tensor]:
    """Optimized tensor conversion - takes direct arguments to avoid Sample object creation."""

    # Time history: reverse to most-recent-first. Raw seconds; encoder log-compresses downstream.
    time_hist_reversed = time_history[::-1].copy()

    # EP mask
    ep_mask = np.zeros(64, dtype=np.float32)
    if 0 <= ep_square < 64:
        ep_mask[ep_square] = 1.0

    # Scalars layout matches _sample_to_tensors. ELO/ply/hmc normalized;
    # time fields (clocks, increments, base_time) are RAW SECONDS — the time
    # encoder picks its own featurization downstream.
    scalars = np.array([
        (active_elo - 1900) / 700.0,
        (opp_elo - 1900) / 700.0,
        move_ply / 100.0,
        float(active_clock),
        float(opp_clock),
        float(active_inc),
        float(opp_inc),
        halfmove_clock / 100.0,
        float(base_time),
    ], dtype=np.float32)

    # Legal move mask - use numpy for speed
    legal_mask = np.zeros(NUM_POLICY_OUTPUTS, dtype=np.bool_)
    for from_sq, to_sq, promo in legal_moves:
        if 0 <= from_sq < 64 and 0 <= to_sq < 64:
            legal_mask[from_sq * 64 + to_sq] = True

    # Policy / promo targets. Resign/flag samples have no move: policy_target is
    # the cross-entropy ignore_index sentinel (-100) and has_move=False.
    promo_target = 0
    promo_file = 0
    has_move = not (policy_is_resign or policy_is_flag)

    if has_move:
        from_sq, to_sq, promo = policy_move
        policy_target_idx = from_sq * 64 + to_sq
        promo_target = promo
        if 56 <= to_sq <= 63:
            promo_file = to_sq - 56
    else:
        policy_target_idx = -100  # ignore_index for cross-entropy

    # Value classification target
    y_val_cls = int(y_val * 2)

    # Time target: absolute seconds at 1s resolution (see _sample_to_tensors).
    time_bin = min(int(round(time_target)), TIME_SECOND_CAP)

    # Convert to tensors
    return {
        "board_history": torch.from_numpy(board_history),
        "time_history": torch.from_numpy(time_hist_reversed),
        "rep_flags": torch.from_numpy(rep_flags.astype(np.float32)),
        "castling": torch.from_numpy(castling.astype(np.float32)),
        "ep_mask": torch.from_numpy(ep_mask),
        "scalars": torch.from_numpy(scalars),
        "legal_mask": torch.from_numpy(legal_mask),
        "policy_target": torch.tensor(policy_target_idx, dtype=torch.long),
        "has_move": torch.tensor(has_move, dtype=torch.bool),
        "resign_target": torch.tensor(float(policy_is_resign), dtype=torch.float32),
        "flag_target": torch.tensor(float(policy_is_flag), dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "y_val_cls": torch.tensor(y_val_cls, dtype=torch.long),
        "time_target_cls": torch.tensor(time_bin, dtype=torch.long),
        "promo_target": torch.tensor(promo_target, dtype=torch.long),
        "promo_file": torch.tensor(promo_file, dtype=torch.long),
        # tc_cat surfaced for per-TC metric tracking (bullet/blitz/rapid/classical/ultraBullet)
        "tc_cat": torch.tensor(tc_cat, dtype=torch.long),
    }


def _make_worker_init_fn(base_seed: int):
    """Return a worker_init_fn that pins each DataLoader worker's RNGs to base_seed + worker_id.

    Default PyTorch worker seeding derives the worker's base_seed from the parent's
    RNG at iterator creation, which means anything in the main process that consumes
    RNG before the iterator starts (e.g., model __init__) shifts the worker seeds.
    For reproducible ablations we pin them explicitly.
    """
    def _init(worker_id: int) -> None:
        seed = (base_seed + worker_id) & 0xFFFFFFFF
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    return _init


def _collate(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a batch of samples - optimized with pre-allocated stacking."""
    if not samples:
        return {}
    keys = samples[0].keys()
    return {
        key: torch.stack([s[key] for s in samples])
        for key in keys
    }


def _normalize_time_history(raw: Sequence[float]) -> np.ndarray:
    """Return an 8-long oldest->newest time history with left zero-padding."""
    time_history = np.array(raw, dtype=np.float32)
    if len(time_history) >= 8:
        return time_history[-8:]
    padded = np.zeros(8, dtype=np.float32)
    if len(time_history) > 0:
        padded[-len(time_history):] = time_history
    return padded


def _infer_base_time(active_clock: float, active_inc: float, base_time: Optional[float] = None) -> float:
    """Recover a best-effort base time for legacy shards that lack the column."""
    if base_time is not None:
        return float(base_time)
    return max(0.0, float(active_clock) + 40.0 * float(active_inc))


class StreamingDataset(IterableDataset):
    """Streaming dataset for parquet data with resume support."""

    def __init__(
        self,
        files: Sequence[str],
        *,
        shuffle_files: bool = True,
        shuffle_rows: bool = True,
        seed: int = 42,
        read_batch_rows: int = 4096,
        filter_tc: Optional[int] = None,
        history_mask_prob: float = 0.0,
    ) -> None:
        self.files = list(files)
        self.shuffle_files = shuffle_files
        self.shuffle_rows = shuffle_rows
        self.seed = seed
        self.read_batch_rows = read_batch_rows
        # With this probability, a sample has a uniformly-random number of its
        # oldest board-history positions zeroed (keeps no-history states
        # in-distribution). Training only — leave 0.0 for validation.
        self.history_mask_prob = history_mask_prob
        # If set, only yield samples whose tc_cat matches this int. Used as a
        # diagnostic (e.g., train-on-blitz-only) without rebuilding the dataset.
        # Throughput drops proportionally to the rejection rate.
        self.filter_tc = filter_tc
        self.epoch = 0
        # Resume support: number of samples to skip at start of epoch
        self._skip_samples = 0
        # Track samples yielded in current epoch (across all workers)
        self._samples_yielded = 0

    def set_epoch(self, epoch: int) -> None:
        # Used to vary shuffling across epochs while staying deterministic.
        self.epoch = int(epoch)
        # Reset skip counter for new epoch
        self._skip_samples = 0
        self._samples_yielded = 0

    def set_skip_samples(self, skip_samples: int) -> None:
        """Set number of samples to skip when iterating (for resume)."""
        self._skip_samples = skip_samples
        self._samples_yielded = skip_samples  # Already counted as yielded

    def get_samples_yielded(self) -> int:
        """Get total samples yielded in current epoch (for checkpointing)."""
        return self._samples_yielded

    def _epoch_seed(self, base_seed: int) -> int:
        # Large odd multiplier to avoid small-cycle patterns.
        return int(base_seed + self.epoch * 1_000_003)

    def _iter_files_for_worker(self) -> Iterable[str]:
        worker = get_worker_info()
        files = self.files
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
            seed = self.seed + worker.id * 17
        else:
            seed = self.seed

        order = list(files)
        if self.shuffle_files:
            rng = random.Random(self._epoch_seed(seed))
            rng.shuffle(order)
        return order

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        # Calculate how many samples this worker should skip
        # Distribute skip evenly across workers
        skip_for_worker = self._skip_samples // num_workers
        if worker_id < (self._skip_samples % num_workers):
            skip_for_worker += 1

        skipped = 0
        yielded = 0

        files_for_worker = list(self._iter_files_for_worker())

        # Fast path: skip entire files by reading their row counts from metadata
        file_idx = 0
        if skip_for_worker > 0:
            for file_idx, file_path in enumerate(files_for_worker):
                try:
                    pf = pq.ParquetFile(file_path)
                    file_rows = pf.metadata.num_rows
                except Exception:
                    file_rows = 0

                if skipped + file_rows <= skip_for_worker:
                    # Skip entire file
                    skipped += file_rows
                else:
                    # This file contains the resume point - break and handle it
                    break
            else:
                # All files skipped (shouldn't happen normally)
                file_idx = len(files_for_worker)

            if skipped > 0:
                # Only log from worker 0 to avoid spam
                if worker_id == 0:
                    print(f"[dataset] Worker {worker_id}: skipped {file_idx} files ({skipped:,} samples), "
                          f"{skip_for_worker - skipped:,} samples remaining to skip in current file")

        # Process remaining files
        for file_path in files_for_worker[file_idx:]:
            for sample in self._stream_file(file_path):
                if skipped < skip_for_worker:
                    skipped += 1
                    continue
                yield sample
                yielded += 1
                # Update samples yielded counter (approximate - not thread-safe but good enough)
                self._samples_yielded = self._skip_samples + (yielded * num_workers)

    def _stream_file(self, file_path: str):
        try:
            parquet_file = pq.ParquetFile(file_path)
        except Exception as exc:
            print(f"[dataset] Failed to open {file_path}: {exc}")
            return

        # Deterministic RNG for this file
        file_hash = int(hashlib.md5(file_path.encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(self._epoch_seed(self.seed) + (file_hash % 1000000))

        try:
            batch_iter = parquet_file.iter_batches(batch_size=self.read_batch_rows)
            while True:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"[dataset] Corrupt batch in {file_path}, skipping rest of file: {e}")
                    break

                batch_len = batch.num_rows
                if batch_len == 0:
                    continue

                # Shuffle rows within batch
                if self.shuffle_rows:
                    shuffle_indices = rng.permutation(batch_len)
                    batch = batch.take(shuffle_indices)

                # Vectorized batch processing - much faster than row-by-row
                try:
                    yield from self._process_batch_vectorized(batch, rng)
                except Exception as e:
                    print(f"[dataset] Error processing batch in {file_path}: {e}")
                    continue
        except Exception as e:
            print(f"[dataset] Failed to iterate {file_path}: {e}")

    def _process_batch_vectorized(self, batch, rng):
        """Process entire batch using vectorized operations instead of row-by-row."""
        # Apply TC filter at the Arrow level before materialising python lists —
        # cheaper than decoding every row then skipping in the row loop.
        if self.filter_tc is not None:
            batch = batch.filter(pc.equal(batch["tc_cat"], self.filter_tc))
            if batch.num_rows == 0:
                return

        batch_len = batch.num_rows

        # Extract all columns at once using to_pylist() - faster than individual .as_py()
        board_history_list = batch["board_history"].to_pylist()
        legal_moves_list = batch["legal_moves"].to_pylist()
        time_history_list = batch["time_history"].to_pylist()
        rep_flags_list = batch["repetition_flags"].to_pylist()
        castling_list = batch["castling_rights"].to_pylist()

        # Scalar columns - convert to numpy arrays for fast access
        ep_square_arr = batch["ep_square"].to_numpy()
        active_elo_arr = batch["active_elo"].to_numpy()
        opp_elo_arr = batch["opp_elo"].to_numpy()
        move_ply_arr = batch["move_ply"].to_numpy()
        halfmove_clock_arr = batch["halfmove_clock"].to_numpy()
        tc_cat_arr = batch["tc_cat"].to_numpy()
        active_clock_arr = batch["active_clock"].to_numpy()
        opp_clock_arr = batch["opp_clock"].to_numpy()
        active_inc_arr = batch["active_inc"].to_numpy()
        opp_inc_arr = batch["opp_inc"].to_numpy()
        # base_time is a 2026-05+ column. Older shards (pre-bullet rebuild) lack it.
        # Reconstruct a best-effort value using the same tc estimate used during
        # preprocessing: T = base + 40 * inc.
        if "base_time" in batch.schema.names:
            base_time_arr = batch["base_time"].to_numpy()
        else:
            base_time_arr = active_clock_arr.astype(np.float32) + 40.0 * active_inc_arr.astype(np.float32)

        # Target columns
        policy_move_list = batch["policy_move"].to_pylist()
        # Boolean columns require zero_copy_only=False
        policy_is_resign_arr = batch["policy_is_resign"].to_numpy(zero_copy_only=False)
        policy_is_flag_arr = batch["policy_is_flag"].to_numpy(zero_copy_only=False)
        y_val_arr = batch["y_val"].to_numpy()
        time_target_arr = batch["time_target"].to_numpy()

        # TC / Elo-band filtering already applied at the Arrow level above.
        for i in range(batch_len):
            try:
                # Board history
                bh = board_history_list[i]
                board_history = np.array([np.array(b) for b in bh], dtype=np.int64)
                if board_history.shape != (8, 64):
                    continue

                # Legal moves
                legal_moves = [(int(m[0]), int(m[1]), int(m[2])) for m in legal_moves_list[i]]

                # Time history
                th = time_history_list[i]
                time_history = _normalize_time_history(th)

                # Rep flags
                rf = rep_flags_list[i]
                rep_flags = np.array(rf, dtype=np.int64) if len(rf) == 8 else np.zeros(8, dtype=np.int64)

                # History masking: with prob history_mask_prob, zero the oldest
                # k history slots. board_history is newest-first, so index 0 is
                # the current position and is never masked. Keeps no-history
                # states in-distribution. time_history is zeroed in lockstep so
                # the per-position clock reconstruction in InputEncoder doesn't
                # leak ply-aged time info at masked positions; when the model is
                # configured without per-position clocks, these slots are unused
                # anyway, so this is a no-op for it.
                if self.history_mask_prob > 0.0 and rng.random() < self.history_mask_prob:
                    hist_len = board_history.shape[0]
                    if hist_len > 1:
                        k = int(rng.integers(1, hist_len))  # 1..hist_len-1
                        board_history[hist_len - k:] = 0
                        rep_flags[hist_len - k:] = 0
                        time_history[hist_len - k:] = 0.0

                # Castling
                cr = castling_list[i]
                castling = np.array(cr, dtype=np.int64) if len(cr) == 4 else np.zeros(4, dtype=np.int64)

                # Policy move
                pm = policy_move_list[i]
                policy_move = (int(pm[0]), int(pm[1]), int(pm[2]))

                yield _sample_to_tensors_fast(
                    board_history=board_history,
                    legal_moves=legal_moves,
                    time_history=time_history,
                    rep_flags=rep_flags,
                    castling=castling,
                    ep_square=int(ep_square_arr[i]),
                    active_elo=int(active_elo_arr[i]),
                    opp_elo=int(opp_elo_arr[i]),
                    move_ply=int(move_ply_arr[i]),
                    halfmove_clock=int(halfmove_clock_arr[i]),
                    tc_cat=int(tc_cat_arr[i]),
                    base_time=float(base_time_arr[i]),
                    active_clock=float(active_clock_arr[i]),
                    opp_clock=float(opp_clock_arr[i]),
                    active_inc=int(active_inc_arr[i]),
                    opp_inc=int(opp_inc_arr[i]),
                    policy_move=policy_move,
                    policy_is_resign=bool(policy_is_resign_arr[i]),
                    policy_is_flag=bool(policy_is_flag_arr[i]),
                    y_val=float(y_val_arr[i]),
                    time_target=float(time_target_arr[i]),
                )
            except Exception:
                continue

    def _extract_sample(self, batch, idx: int) -> Optional[Sample]:
        """Extract a single sample from a batch at the given index."""
        try:
            # Board history: nested array (8, 64)
            board_history_raw = batch["board_history"][idx].as_py()
            board_history = np.array([np.array(b) for b in board_history_raw], dtype=np.int64)
            if board_history.shape != (8, 64):
                return None

            # Legal moves: list of [from, to, promo]
            legal_moves_raw = batch["legal_moves"][idx].as_py()
            legal_moves = [(int(m[0]), int(m[1]), int(m[2])) for m in legal_moves_raw]

            # Time history: (8,)
            time_history = _normalize_time_history(batch["time_history"][idx].as_py())

            # Repetition flags: (8,)
            rep_flags = np.array(batch["repetition_flags"][idx].as_py(), dtype=np.int64)
            if len(rep_flags) != 8:
                rep_flags = np.zeros(8, dtype=np.int64)

            # Castling rights: (4,)
            castling = np.array(batch["castling_rights"][idx].as_py(), dtype=np.int64)
            if len(castling) != 4:
                castling = np.zeros(4, dtype=np.int64)

            # Scalars
            ep_square = int(batch["ep_square"][idx].as_py())
            active_elo = int(batch["active_elo"][idx].as_py())
            opp_elo = int(batch["opp_elo"][idx].as_py())
            move_ply = int(batch["move_ply"][idx].as_py())
            halfmove_clock = int(batch["halfmove_clock"][idx].as_py())
            tc_cat = int(batch["tc_cat"][idx].as_py())
            active_clock = float(batch["active_clock"][idx].as_py())
            opp_clock = float(batch["opp_clock"][idx].as_py())
            active_inc = int(batch["active_inc"][idx].as_py())
            opp_inc = int(batch["opp_inc"][idx].as_py())
            if "base_time" in batch.schema.names:
                base_time = float(batch["base_time"][idx].as_py())
            else:
                base_time = _infer_base_time(active_clock, active_inc)

            # Targets
            policy_move_raw = batch["policy_move"][idx].as_py()
            policy_move = (int(policy_move_raw[0]), int(policy_move_raw[1]), int(policy_move_raw[2]))
            policy_is_resign = bool(batch["policy_is_resign"][idx].as_py())
            policy_is_flag = bool(batch["policy_is_flag"][idx].as_py())
            y_val = float(batch["y_val"][idx].as_py())
            time_target = float(batch["time_target"][idx].as_py())

            return Sample(
                board_history=board_history,
                legal_moves=legal_moves,
                time_history=time_history,
                repetition_flags=rep_flags,
                castling_rights=castling,
                ep_square=ep_square,
                active_elo=active_elo,
                opp_elo=opp_elo,
                move_ply=move_ply,
                halfmove_clock=halfmove_clock,
                tc_cat=tc_cat,
                base_time=base_time,
                active_clock=active_clock,
                opp_clock=opp_clock,
                active_inc=active_inc,
                opp_inc=opp_inc,
                policy_move=policy_move,
                policy_is_resign=policy_is_resign,
                policy_is_flag=policy_is_flag,
                y_val=y_val,
                time_target=time_target,
            )
        except Exception:
            return None


def _split_files_deterministic(files: List[Path], val_fraction: float = 0.05,
                                max_val_files: int = 1) -> Tuple[List[Path], List[Path]]:
    """Split files into train/val using deterministic hash-based assignment.
    
    Each file is assigned to train or val based on the MD5 hash of its name.
    This ensures:
    1. Repeatability - same files always go to same split
    2. No data leakage - files are never in both splits
    3. Approximate balance - hash distribution is uniform
    
    Args:
        files: List of parquet files to split
        val_fraction: Fraction of files to use for validation (default 5%)
        max_val_files: Maximum number of files in the val split (default 1).
                       Extra files that would hash to val are moved to train.
    
    Returns:
        (train_files, val_files) tuple
    """
    train_files = []
    val_files = []

    for f in files:
        # Hash the filename (not full path) for determinism
        file_hash = int(hashlib.md5(f.name.encode('utf-8')).hexdigest(), 16)
        # Use modulo to get a value in [0, 1)
        hash_val = (file_hash % 100000) / 100000.0

        if hash_val < val_fraction and len(val_files) < max_val_files:
            val_files.append(f)
        else:
            train_files.append(f)

    return train_files, val_files


def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    shuffle: bool = None,
    pin_memory: bool = True,
    prefetch_factor: int = None,
    val_fraction: float = 0.05,
    max_val_files: int = 1,
    persistent_workers: bool = True,
    drop_last: bool = True,
    filter_tc: Optional[int] = None,
    history_mask_prob: float = 0.0,
) -> DataLoader:
    """Create a single dataloader for a split.

    Args:
        data_dir: Root directory containing train/ subdirectory
        split: 'train' | 'val' | 'all'.
            'train' / 'val' apply hash-based file splitting from data_dir/train/.
            'all' uses every parquet file in data_dir/train/ with no splitting
            (use this for a held-out val directory so no shards are silently dropped).
        val_fraction: Fraction of files to use for validation (default 5%). Ignored when split='all'.
        max_val_files: Maximum number of files in the val split (default 1). Ignored when split='all'.
        persistent_workers: Keep workers alive between epochs (faster)
        drop_last: Drop last incomplete batch (better for training)
        Other args: standard DataLoader configuration
    """

    # Always load from train directory
    train_dir = Path(data_dir) / "train"
    all_files = sorted(train_dir.rglob("*.parquet"))

    if len(all_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {train_dir}")

    if split == "all":
        files = all_files
    else:
        # Split files deterministically
        train_files, val_files = _split_files_deterministic(all_files, val_fraction, max_val_files)
        if split == "train":
            files = train_files
        elif split == "val":
            files = val_files
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'all'.")

    print(f"[dataset] Split: {split} | Files: {len(files)}/{len(all_files)} ({len(files)/len(all_files)*100:.1f}%)")

    is_train = split == "train"
    should_shuffle = shuffle if shuffle is not None else is_train

    ds = StreamingDataset(
        [str(f) for f in files],
        shuffle_files=should_shuffle,
        shuffle_rows=should_shuffle,
        seed=seed,
        filter_tc=filter_tc,
        # History masking is a training-only augmentation.
        history_mask_prob=history_mask_prob if is_train else 0.0,
    )
    if filter_tc is not None:
        print(f"[dataset] filter_tc={filter_tc} — only yielding samples with this tc_cat")

    # Use persistent_workers to avoid worker restart overhead between epochs
    use_persistent = persistent_workers and num_workers > 0

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=use_persistent,
        drop_last=drop_last if is_train else False,
        generator=generator,
        worker_init_fn=_make_worker_init_fn(seed),
    )


def count_samples(data_dir: str, split: str = "train", val_fraction: float = 0.05, max_val_files: int = 1) -> int:
    """Estimate number of samples in a split by reading parquet metadata.

    Mirrors create_dataloader's split logic, including the 'all' mode.
    """
    train_dir = Path(data_dir) / "train"
    all_files = sorted(train_dir.rglob("*.parquet"))

    if split == "all":
        files = all_files
    else:
        train_files, val_files = _split_files_deterministic(all_files, val_fraction, max_val_files)
        if split == "train":
            files = train_files
        elif split == "val":
            files = val_files
        else:
            raise ValueError(f"Unknown split: {split}")

    total = 0
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            total += pf.metadata.num_rows
        except Exception:
            pass

    return total


