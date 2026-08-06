#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_dataset.config import (  # noqa: E402
    BALANCED_DIR,
    BALANCED_TC_CATS,
    DEFAULT_TARGET_PER_BIN,
    ELO_MAX,
    ELO_MIN,
    ELO_STEP,
    EXTRA_TC_CATS,
    TC_NAMES,
    extra_tc_targets,
    target_per_bucket,
)


def _build_state_path() -> Path:
    return BALANCED_DIR / "build_state.json"


def _configured_target_per_bin() -> int:
    path = _build_state_path()
    if not path.exists():
        return DEFAULT_TARGET_PER_BIN
    try:
        with open(path) as f:
            state = json.load(f)
        return int(state.get("config", {}).get("target_per_bin", DEFAULT_TARGET_PER_BIN))
    except Exception:
        return DEFAULT_TARGET_PER_BIN


def _elo_bin(elo: int) -> int | None:
    if elo < ELO_MIN or elo >= ELO_MAX:
        return None
    return (elo - ELO_MIN) // ELO_STEP


def _month_dirs(train_dir: Path, month: str | None, include_all: bool) -> list[Path]:
    excluded = {"shuffled", "_shuffle_partitions"}
    dirs = [p for p in train_dir.iterdir() if p.is_dir() and p.name not in excluded]
    if month is not None:
        out = []
        for name in (month, f"{month}.tmp"):
            p = train_dir / name
            if p.is_dir():
                out.append(p)
        return sorted(out)
    if include_all:
        return sorted(dirs)
    return sorted(dirs)


def _parquet_files(dirs: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in dirs:
        out.extend(sorted(path.glob("balanced_*.parquet")))
    return out


def scan_counts(files: list[Path]) -> tuple[int, dict[tuple[int, int], int], dict[int, int]]:
    import pyarrow.parquet as pq

    counts: dict[tuple[int, int], int] = defaultdict(int)
    oor: dict[int, int] = defaultdict(int)
    rows = 0
    for shard in files:
        pf = pq.ParquetFile(shard)
        rows += pf.metadata.num_rows
        for batch in pf.iter_batches(batch_size=65536, columns=["active_elo", "tc_cat"]):
            elos = batch.column("active_elo").to_numpy()
            tcs = batch.column("tc_cat").to_numpy()
            for elo, tc in zip(elos.tolist(), tcs.tolist()):
                tc = int(tc)
                b = _elo_bin(int(elo))
                if b is None:
                    oor[tc] += 1
                else:
                    counts[(b, tc)] += 1
    return rows, counts, oor


def print_report(dirs: list[Path], files: list[Path], rows: int,
                 counts: dict[tuple[int, int], int], oor: dict[int, int],
                 target_per_bin: int) -> None:
    core_targets = target_per_bucket(target_per_bin)
    extra_targets = extra_tc_targets(target_per_bin)
    num_bins = (ELO_MAX - ELO_MIN) // ELO_STEP

    print("Dirs:")
    for path in dirs:
        print(f"  {path.relative_to(REPO_ROOT)}")
    print(f"Files: {len(files):,}")
    print(f"Rows:  {rows:,}")
    print()

    print("Per Cell Progress")
    for b in range(num_bins):
        lo = ELO_MIN + b * ELO_STEP
        hi = lo + ELO_STEP - 1
        parts = []
        for tc in BALANCED_TC_CATS:
            count = counts[(b, tc)]
            target = core_targets[tc]
            pct = 100.0 * count / max(1, target)
            parts.append(f"{TC_NAMES[tc]}: {count:,} ({pct:.1f}%)")
        print(f"{lo}-{hi}: " + " | ".join(parts))
    print()

    print("Core Totals")
    for tc in BALANCED_TC_CATS:
        total = sum(counts[(b, tc)] for b in range(num_bins))
        target = core_targets[tc] * num_bins
        pct = 100.0 * total / max(1, target)
        print(f"{TC_NAMES[tc]}: {total:,} / {target:,} ({pct:.1f}%)")
    print()

    print("Extra Time Controls")
    for tc in EXTRA_TC_CATS:
        in_range = sum(counts[(b, tc)] for b in range(num_bins))
        target = extra_targets[tc]
        pct = 100.0 * in_range / max(1, target)
        print(f"{TC_NAMES[tc]}: in-range {in_range:,} / {target:,} ({pct:.1f}%), oor {oor.get(tc, 0):,}")
    print()

    print("Out Of Range")
    for tc in range(len(TC_NAMES)):
        if oor.get(tc, 0):
            print(f"{TC_NAMES[tc]}: {oor[tc]:,}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="build_dataset_stats")
    parser.add_argument("--month", default=None,
                        help="Specific month label YYYY-MM to inspect. Includes both month/ and month.tmp/ if present.")
    parser.add_argument("--all", action="store_true",
                        help="Scan all month directories under data/train/train/. This is the default behavior.")
    parser.add_argument("--target-per-bin", type=int, default=None,
                        help="Override the target_per_bin used for progress percentages.")
    args = parser.parse_args()

    train_dir = BALANCED_DIR / "train"
    if not train_dir.exists():
        print(f"No train directory at {train_dir}")
        return 1

    dirs = _month_dirs(train_dir, args.month, include_all=args.all)
    files = _parquet_files(dirs)
    if not files:
        print("No parquet shards found for the requested scope.")
        return 1

    target_per_bin = args.target_per_bin or _configured_target_per_bin()
    rows, counts, oor = scan_counts(files)
    print_report(dirs, files, rows, counts, oor, target_per_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
