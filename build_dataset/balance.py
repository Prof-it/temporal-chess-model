"""Final balance pass.

Reads all stage/*/stage_*.parquet shards, computes keep_prob per (elo_bin,
tc_cat) so each in-range bucket lands at its target, and writes shuffled
balanced_NNNNN.parquet shards under data/train/train/.

Out-of-range rows (in_balance_range=False) always pass.
"""
from __future__ import annotations

import gc
import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    CHUNK_SIZE,
    DEFAULT_SEED,
    NUM_ELO_BINS,
    NUM_TC_CATS,
    STAGE_DIR,
    TRAIN_DIR,
    target_per_bucket,
)


def row_keep_hash(game_id: str, ply: int, seed: int) -> float:
    """Deterministic [0,1) hash for a (game_id, ply) pair. Lifted from build_balanced_dataset.py."""
    raw = f"{game_id}|{ply}|{seed}".encode("utf-8")
    h = hashlib.blake2s(raw, digest_size=8).digest()
    return int.from_bytes(h, "little") / (2**64)


def scan_stage_counts(stage_dir: Path = STAGE_DIR) -> Tuple[Dict[Tuple[int, int], int], int, int]:
    """Pass 1: aggregate per-(elo_bin, tc_cat) counts across all stage shards.

    Returns (counts dict, total_in_range, total_out_of_range).
    """
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    total_in = 0
    total_out = 0
    shards = sorted(stage_dir.glob("*/stage_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No stage shards under {stage_dir}")
    print(f"[balance/pass1] scanning {len(shards)} stage shards…", flush=True)
    t0 = time.time()
    for i, s in enumerate(shards):
        pf = pq.ParquetFile(s)
        for batch in pf.iter_batches(batch_size=65536,
                                     columns=["active_elo", "tc_cat", "in_balance_range"]):
            elos = batch.column("active_elo").to_numpy()
            tcs = batch.column("tc_cat").to_numpy()
            in_rng = batch.column("in_balance_range").to_numpy(zero_copy_only=False)
            in_mask = in_rng.astype(bool)
            total_in += int(in_mask.sum())
            total_out += int((~in_mask).sum())
            # Vectorized count by (elo_bin, tc_cat)
            in_elos = elos[in_mask]
            in_tcs = tcs[in_mask]
            bins = ((in_elos - 1200) // 100).astype(np.int32)
            for b, t in zip(bins.tolist(), in_tcs.tolist()):
                counts[(int(b), int(t))] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(shards)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[balance/pass1] in_range={total_in:,} oor={total_out:,} elapsed={time.time()-t0:.0f}s",
          flush=True)
    return counts, total_in, total_out


def compute_keep_probs(
    counts: Dict[Tuple[int, int], int],
    target_per_bin: int,
) -> Dict[Tuple[int, int], float]:
    """For each (elo_bin, tc_cat) compute keep_prob = min(1, target / count)."""
    targets = target_per_bucket(target_per_bin)
    keep_prob: Dict[Tuple[int, int], float] = {}
    for b in range(NUM_ELO_BINS):
        for t in range(NUM_TC_CATS):
            n = counts.get((b, t), 0)
            tgt = targets[t]
            keep_prob[(b, t)] = min(1.0, tgt / n) if n > 0 else 0.0
    return keep_prob


def write_balanced_shards(
    keep_prob: Dict[Tuple[int, int], float],
    seed: int = DEFAULT_SEED,
    stage_dir: Path = STAGE_DIR,
    out_dir: Path = TRAIN_DIR,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[int, int, Dict[Tuple[int, int], int], int]:
    """Pass 2: stream stage shards, apply per-row keep, write shuffled output.

    Returns (rows_written, shards_written, final_in_range_counts, final_oor_count).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(stage_dir.glob("*/stage_*.parquet"))

    rng = np.random.default_rng(seed)
    buffer_tables = []
    buffered_rows = 0
    chunk_idx = 0
    total_rows = 0
    final_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    final_oor = 0

    print(f"[balance/pass2] writing balanced shards from {len(shards)} stage shards…",
          flush=True)
    t0 = time.time()

    def flush_chunk() -> int:
        nonlocal buffer_tables, buffered_rows, chunk_idx, total_rows
        if buffered_rows == 0:
            return 0
        combined = pa.concat_tables(buffer_tables)
        # Take exactly chunk_size rows (or fewer at very end).
        take_n = min(chunk_size, combined.num_rows)
        idxs = rng.permutation(combined.num_rows)[:take_n]
        out_table = combined.take(pa.array(idxs))
        out_path = out_dir / f"balanced_{chunk_idx:05d}.parquet"
        pq.write_table(out_table, out_path, compression="zstd")
        chunk_idx += 1
        total_rows += out_table.num_rows
        # Anything left over carries forward.
        if combined.num_rows > take_n:
            mask = np.ones(combined.num_rows, dtype=bool)
            mask[idxs] = False
            remainder_idx = np.where(mask)[0]
            buffer_tables = [combined.take(pa.array(remainder_idx))]
            buffered_rows = int(remainder_idx.size)
        else:
            buffer_tables = []
            buffered_rows = 0
        del combined, out_table
        gc.collect()
        return take_n

    for i, s in enumerate(shards):
        table = pq.read_table(s)
        n = table.num_rows
        elos = table.column("active_elo").to_numpy()
        tcs = table.column("tc_cat").to_numpy()
        in_rng = table.column("in_balance_range").to_numpy(zero_copy_only=False).astype(bool)
        game_ids = table.column("game_id").to_pylist()
        plies = table.column("move_ply").to_numpy()

        bins = ((elos - 1200) // 100).astype(np.int32)
        # Per-row keep mask
        keep = np.zeros(n, dtype=bool)
        for j in range(n):
            if not in_rng[j]:
                keep[j] = True  # always keep out-of-range
                continue
            prob = keep_prob.get((int(bins[j]), int(tcs[j])), 0.0)
            if prob >= 1.0:
                keep[j] = True
            elif prob > 0.0:
                h = row_keep_hash(str(game_ids[j]), int(plies[j]), seed)
                keep[j] = h < prob

        if keep.any():
            kept_idx = np.where(keep)[0]
            kept = table.take(pa.array(kept_idx))
            buffer_tables.append(kept)
            buffered_rows += int(kept_idx.size)
            # Tally
            for j in kept_idx.tolist():
                if in_rng[j]:
                    final_counts[(int(bins[j]), int(tcs[j]))] += 1
                else:
                    final_oor += 1

            while buffered_rows >= chunk_size:
                flush_chunk()

        if (i + 1) % 50 == 0:
            print(f"  pass2 {i+1}/{len(shards)} chunks={chunk_idx} rows={total_rows:,} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        del table

    # Final flush of remainder
    while buffered_rows > 0:
        flush_chunk()

    print(f"[balance/pass2] done. shards={chunk_idx} rows={total_rows:,} "
          f"oor={final_oor:,} elapsed={time.time()-t0:.0f}s", flush=True)
    return total_rows, chunk_idx, dict(final_counts), final_oor


def run_balance_pass(target_per_bin: int, seed: int = DEFAULT_SEED) -> dict:
    """End-to-end balance pass. Returns a summary dict for the manifest."""
    counts, total_in, total_out = scan_stage_counts()
    keep_prob = compute_keep_probs(counts, target_per_bin)
    rows, shards, final_counts, final_oor = write_balanced_shards(keep_prob, seed=seed)
    return {
        "stage_in_range": total_in,
        "stage_out_of_range": total_out,
        "stage_counts": {f"{b}_{t}": c for (b, t), c in counts.items()},
        "keep_prob": {f"{b}_{t}": p for (b, t), p in keep_prob.items()},
        "final_counts": {f"{b}_{t}": c for (b, t), c in final_counts.items()},
        "final_out_of_range": final_oor,
        "balanced_total": rows,
        "num_shards": shards,
    }
