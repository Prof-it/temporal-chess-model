"""Pipeline-wide constants. Single source of truth for thresholds, paths, schema.

To redirect output to a different directory (e.g. for smoke testing), set the
MARVIN_DATA_DIR environment variable BEFORE importing this module:
    MARVIN_DATA_DIR=/tmp/smoke python3 -m build_dataset ...
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MARVIN_DATA_DIR", str(REPO_ROOT / "data"))).resolve()
# The build writes here; train.py is then pointed at it with --data-dir.
# dataset.py resolves shards as <data-dir>/train/**/*.parquet, which is why the
# dataset root contains a nested train/ directory.
BALANCED_DIR = DATA_DIR / "train"
TRAIN_DIR = BALANCED_DIR / "train"
STAGE_DIR = BALANCED_DIR / "stage"
STATE_FILE = BALANCED_DIR / "build_state.json"
MANIFEST_FILE = BALANCED_DIR / "manifest.json"
VERIFY_REPORT = BALANCED_DIR / "verify_report.json"

LICHESS_URL_TEMPLATE = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)

# ELO bins: 13 bins of 100 ELO covering [1200, 2500). Out-of-range rows are kept
# but not balanced; they get in_balance_range=False.
ELO_MIN = 1200
ELO_MAX = 2500
ELO_STEP = 100
NUM_ELO_BINS = (ELO_MAX - ELO_MIN) // ELO_STEP  # 13

# Time control categories matching Lichess perfType naming.
# Estimated total seconds T = base + 40 * inc.
#   T <   30  → 4 ultraBullet
#   T <  180  → 0 bullet
#   T <  480  → 1 blitz
#   T < 1500  → 2 rapid
#   T >= 1500 → 3 classical
ULTRABULLET_MAX = 30
TC_THRESHOLDS = [180, 480, 1500]  # upper bound of bullet, blitz, rapid
TC_NAMES = ("bullet", "blitz", "rapid", "classical", "ultraBullet")
NUM_TC_CATS = 5
TC_BULLET, TC_BLITZ, TC_RAPID, TC_CLASSICAL, TC_ULTRABULLET = 0, 1, 2, 3, 4

# Per-bin position targets. 100M per ELO bin, split evenly across the 3
# balanced TCs (bullet, blitz, rapid). Classical is kept but NOT balanced —
# it's supply-limited at high ELO and doesn't fit the uniform-target frame.
DEFAULT_TARGET_PER_BIN = 120_000_000
TC_FRACTIONS = {
    TC_BULLET: 1.0 / 3.0,
    TC_BLITZ: 1.0 / 3.0,
    TC_RAPID: 1.0 / 3.0,
}
# Classical and ultraBullet are intentionally absent from TC_FRACTIONS —
# they are kept as low-priority capped extras rather than balanced targets.
BALANCED_TC_CATS = (TC_BULLET, TC_BLITZ, TC_RAPID)
EXTRA_TC_CAP_FRACTIONS = {
    TC_CLASSICAL: 0.05,
    TC_ULTRABULLET: 0.01,
}
EXTRA_TC_CATS = tuple(EXTRA_TC_CAP_FRACTIONS)

# Out-of-range ELO (active_elo not in [ELO_MIN, ELO_MAX)) is kept but capped
# at this fraction of the balanced total so it doesn't dominate the dataset.
# E.g. with target_per_bin=75M → balanced ≈ 975M → OOR target ≈ 97.5M.
OOR_FRACTION = 0.10


def oor_target(target_per_bin: int = DEFAULT_TARGET_PER_BIN) -> int:
    """Total OOR row target = balanced_total × OOR_FRACTION."""
    return int(target_per_bin * NUM_ELO_BINS * OOR_FRACTION)


def extra_tc_targets(target_per_bin: int = DEFAULT_TARGET_PER_BIN) -> dict:
    """Global caps for low-priority time controls, outside the core balance goal."""
    balanced_total = target_per_bin * NUM_ELO_BINS
    return {
        tc: int(balanced_total * frac)
        for tc, frac in EXTRA_TC_CAP_FRACTIONS.items()
    }

# Headroom multiplier on per-month per-bucket cap (gives the balance pass slack).
PER_MONTH_CAP_HEADROOM = 1.5
PER_MONTH_CAP_FLOOR = 100_000  # never cap below this even for tiny month counts

CHUNK_SIZE = 250_000  # rows per parquet shard
HISTORY_LEN = 8       # board history length (matches process_pgn_balanced.py)

# Producer/consumer queue sizes
GAME_QUEUE_MAXSIZE = 1000   # PGN game text blocks waiting for workers
ROW_QUEUE_MAXSIZE = 200     # batches of rows waiting for the writer
ROW_BATCH_SIZE = 1000       # rows per batch from worker → writer

DEFAULT_WORKERS = 8
DEFAULT_SEED = 20260509

# Game-level filter (matches existing pipeline; rejects games where neither
# player has a useful rating).
GAME_ELO_MIN = 800

# HTTP retry behavior
HTTP_TIMEOUT_CONNECT = 30   # seconds
HTTP_TIMEOUT_READ = 300     # seconds (read timeout, not total)
HTTP_RETRIES = 3
HTTP_BACKOFF_BASE = 5.0


def elo_bin(elo: int) -> int:
    """Return ELO bin index in [0, NUM_ELO_BINS) or -1 if out of range."""
    if elo < ELO_MIN or elo >= ELO_MAX:
        return -1
    return (elo - ELO_MIN) // ELO_STEP


def bin_label(b: int) -> str:
    lo = ELO_MIN + b * ELO_STEP
    return f"{lo}–{lo + ELO_STEP}"


def get_tc_category(base: int, inc: int) -> int:
    """Return tc_cat in [0, 5), including capped ultraBullet rows."""
    t = base + 40 * inc
    if t < ULTRABULLET_MAX:
        return TC_ULTRABULLET
    if t < TC_THRESHOLDS[0]:
        return TC_BULLET
    if t < TC_THRESHOLDS[1]:
        return TC_BLITZ
    if t < TC_THRESHOLDS[2]:
        return TC_RAPID
    return TC_CLASSICAL


def target_per_bucket(target_per_bin: int = DEFAULT_TARGET_PER_BIN) -> dict:
    """Per-(tc_cat) absolute target derived from per-bin target × tc_fraction."""
    return {tc: int(target_per_bin * frac) for tc, frac in TC_FRACTIONS.items()}


def per_month_cap(target_per_bucket_val: int, num_months: int) -> int:
    """How many positions per (elo_bin, tc_cat) to emit per month before capping."""
    cap = int(target_per_bucket_val * PER_MONTH_CAP_HEADROOM / max(num_months, 1))
    return max(cap, PER_MONTH_CAP_FLOOR)
