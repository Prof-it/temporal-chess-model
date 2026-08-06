"""Evaluation dataset loading and batching.

Resolves an eval set to parquet shards, streams rows, and yields model-ready
batches (via :func:`inference.encoding.make_model_batch_from_parquet_row`)
paired with per-row metadata used by the metrics layer (Elo, clock ratio,
time-pressure bucket, ground-truth move / outcome / ponder time, resign flag).

Time-pressure buckets follow the thesis definition (ratio of remaining clock to
the time-control base): comfortable >50%, moderate 20-50%, pressured 5-20%,
critical <5%. The current parquet schema carries ``base_time`` directly, so the
ratio needs no game-level proxy estimation.

Time-head decoding uses the *current* scheme (model.NUM_TIME_BINS bins where the
bin index is seconds-spent, capped at TIME_SECOND_CAP) — not the legacy
sqrt-ratio-over-128 scheme some older scripts assume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from inference.encoding import make_model_batch_from_parquet_row

# Time-pressure buckets (clock remaining as fraction of base time).
BUCKET_EDGES = (0.5, 0.2, 0.05)          # > / in / in / < boundaries
BUCKET_LABELS = (">50%", "20-50%", "5-20%", "<5%")
BUCKET_NAMES = ("Comfortable", "Moderate", "Pressured", "Critical")

RATING_BANDS = ((1200, 1500), (1500, 1800), (1800, 2100), (2100, 2500))

_BATCH_KEYS = (
    "board_history", "time_history", "rep_flags", "castling",
    "ep_mask", "scalars", "legal_mask",
)


def bucket_index(clock_ratio: float) -> int:
    """Map a clock-remaining ratio to a time-pressure bucket index (0..3)."""
    if clock_ratio > BUCKET_EDGES[0]:
        return 0
    if clock_ratio > BUCKET_EDGES[1]:
        return 1
    if clock_ratio > BUCKET_EDGES[2]:
        return 2
    return 3


def rating_band_index(elo: float) -> int:
    """Index into RATING_BANDS, or -1 if outside the covered range."""
    for i, (lo, hi) in enumerate(RATING_BANDS):
        if lo <= elo < hi:
            return i
    return -1


@dataclass
class RowMeta:
    """Per-row metadata aligned with a model batch, for the metrics layer."""
    active_elo: int
    opp_elo: int
    clock_ratio: float
    bucket_index: int
    rating_band: int
    base_time_s: float
    active_clock_s: float
    legal_moves: list           # list of [from, to, promo]
    policy_move: list           # [from, to, promo]
    target_move_idx: int        # from*64 + to (regular moves)
    y_val: int                  # 0=loss, 1=draw, 2=win  (active perspective)
    time_target_s: float        # seconds spent on the move (ground truth)
    is_resign: bool
    is_flag: bool


@dataclass
class EvalBatch:
    inputs: dict[str, np.ndarray]    # stacked model inputs, batch-first
    metas: list[RowMeta]


def resolve_eval_files(spec: str | Path) -> tuple[list[Path], str]:
    """Resolve an eval-set spec to parquet files + a human label.

    ``spec`` may be a directory of ``*.parquet`` (searched, then ``train/``
    subdir) or a single ``.parquet`` file.
    """
    p = Path(spec)
    if p.is_file() and p.suffix == ".parquet":
        return [p], p.stem
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            files = sorted((p / "train").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files under {p}")
        return files, p.name
    raise FileNotFoundError(f"Eval set not found: {spec}")


def resolve_base_time(row: dict) -> float:
    """True game base time in seconds. Train shards store ``base_time``; test/
    analysis shards store ``tc_base_s``. Never fall back to a clock-derived proxy
    for the *model input* — that silently corrupts the time-conditioning signal.
    """
    for key in ("base_time", "tc_base_s"):
        v = row.get(key)
        if v is not None and float(v) > 0:
            return float(v)
    # Last resort only for bucketing when neither column exists.
    return max(float(row["active_clock"]), float(row["opp_clock"]), 1.0)


def _row_to_meta(row: dict) -> RowMeta:
    base_time = resolve_base_time(row)
    active_clock = float(row["active_clock"])
    ratio = min(max(active_clock / max(base_time, 1.0), 0.0), 1.0)
    pm = list(row["policy_move"])
    return RowMeta(
        active_elo=int(row["active_elo"]),
        opp_elo=int(row["opp_elo"]),
        clock_ratio=ratio,
        bucket_index=bucket_index(ratio),
        rating_band=rating_band_index(int(row["active_elo"])),
        base_time_s=base_time,
        active_clock_s=active_clock,
        legal_moves=[list(m) for m in row["legal_moves"]],
        policy_move=pm,
        target_move_idx=int(pm[0]) * 64 + int(pm[1]),
        # parquet y_val is the float outcome 0.0/0.5/1.0 (loss/draw/win); the
        # WDL class index is y_val*2 -> 0/1/2 (matches dataset.py training target).
        y_val=int(round(float(row["y_val"]) * 2)),
        time_target_s=float(row["time_target"]),
        is_resign=bool(row["policy_is_resign"]),
        is_flag=bool(row["policy_is_flag"]),
    )


def _encode_row(row: dict) -> dict[str, np.ndarray]:
    batch = make_model_batch_from_parquet_row(
        board_history=row["board_history"],
        legal_moves=row["legal_moves"],
        repetition_flags=row["repetition_flags"],
        time_history_s=row["time_history"],
        castling_rights=row["castling_rights"],
        ep_square=row["ep_square"],
        active_elo=row["active_elo"],
        opponent_elo=row["opp_elo"],
        move_ply=row["move_ply"],
        halfmove_clock=row["halfmove_clock"],
        active_clock_s=row["active_clock"],
        opponent_clock_s=row["opp_clock"],
        active_inc_s=row["active_inc"],
        opponent_inc_s=row["opp_inc"],
        base_time_s=resolve_base_time(row),
    )
    # make_model_batch returns a leading batch dim of 1; squeeze it for stacking.
    return {k: np.asarray(batch[k])[0] for k in _BATCH_KEYS}


def iter_eval_batches(
    files: list[Path],
    batch_size: int = 384,
    limit: int | None = None,
    progress: bool = True,
) -> Iterator[EvalBatch]:
    """Stream parquet rows from ``files`` as encoded model batches + metadata."""
    columns = [
        "board_history", "legal_moves", "time_history", "active_elo", "opp_elo",
        "move_ply", "halfmove_clock", "repetition_flags", "castling_rights",
        "ep_square", "base_time", "tc_base_s", "active_clock", "opp_clock", "active_inc",
        "opp_inc", "y_val", "policy_move", "policy_is_resign", "policy_is_flag",
        "time_target",
    ]
    buf_inputs: dict[str, list] = {k: [] for k in _BATCH_KEYS}
    buf_metas: list[RowMeta] = []
    seen = 0

    iterator = files
    if progress:
        from tqdm import tqdm
        iterator = tqdm(files, desc="eval shards", unit="file")

    def _flush() -> EvalBatch:
        inputs = {k: np.stack(buf_inputs[k]) for k in _BATCH_KEYS}
        metas = list(buf_metas)
        for k in _BATCH_KEYS:
            buf_inputs[k].clear()
        buf_metas.clear()
        return EvalBatch(inputs=inputs, metas=metas)

    for parquet_path in iterator:
        pf = pq.ParquetFile(str(parquet_path))
        avail = set(pf.schema_arrow.names)
        cols = [c for c in columns if c in avail]
        for record_batch in pf.iter_batches(batch_size=50_000, columns=cols):
            for row in record_batch.to_pylist():
                buf_inputs_row = _encode_row(row)
                for k in _BATCH_KEYS:
                    buf_inputs[k].append(buf_inputs_row[k])
                buf_metas.append(_row_to_meta(row))
                seen += 1
                if len(buf_metas) >= batch_size:
                    yield _flush()
                if limit is not None and seen >= limit:
                    if buf_metas:
                        yield _flush()
                    return
    if buf_metas:
        yield _flush()
