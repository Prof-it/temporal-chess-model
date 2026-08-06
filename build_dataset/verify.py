"""End-to-end verification of the built dataset.

Checks (writes data/train/verify_report.json):
  1. Counts: per-(elo_bin, tc_cat) within 1% of target; OOR bucket non-zero.
  2. TC consistency: tc_cat == get_tc_category(base_time, active_inc) on a sample.
  3. Schema present in random shards.
  4. Loadable: import dataset and pull batches without exception.
  5. OOR sanity: in_balance_range=False rows have active_elo outside [1200,2500).
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow.parquet as pq

from .config import (
    DEFAULT_TARGET_PER_BIN,
    ELO_MAX,
    ELO_MIN,
    EXTRA_TC_CATS,
    NUM_ELO_BINS,
    NUM_TC_CATS,
    TC_NAMES,
    TRAIN_DIR,
    VERIFY_REPORT,
    get_tc_category,
    extra_tc_targets,
    target_per_bucket,
)

EXPECTED_COLUMNS = {
    "game_id", "split", "board_history", "legal_moves", "time_history",
    "active_elo", "opp_elo", "move_ply", "halfmove_clock", "repetition_flags",
    "castling_rights", "ep_square", "tc_cat", "base_time",
    "active_clock", "opp_clock", "active_inc", "opp_inc",
    "y_val", "policy_move", "policy_is_resign", "policy_is_flag",
    "time_target", "in_balance_range",
}


def _verified_shards(train_dir: Path = TRAIN_DIR) -> list:
    """Return committed train shards, excluding scratch/output directories."""
    excluded_dirs = {"shuffled", "_shuffle_partitions"}
    return sorted(
        s for s in train_dir.rglob("balanced_*.parquet")
        if not any(part in excluded_dirs or part.endswith(".tmp") for part in s.parts)
    )


def check_schema(shards: list) -> Tuple[bool, list]:
    sample = random.sample(shards, min(3, len(shards)))
    issues = []
    for s in sample:
        names = set(pq.read_schema(s).names)
        missing = EXPECTED_COLUMNS - names
        if missing:
            issues.append(f"{s.name}: missing {sorted(missing)}")
    return not issues, issues


def check_counts(shards: list, target_per_bin: int) -> Tuple[bool, dict]:
    targets = target_per_bucket(target_per_bin)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    oor = 0
    print(f"[verify/counts] scanning {len(shards)} shards…", flush=True)
    t0 = time.time()
    for i, s in enumerate(shards):
        pf = pq.ParquetFile(s)
        for batch in pf.iter_batches(batch_size=65536,
                                     columns=["active_elo", "tc_cat", "in_balance_range"]):
            elos = batch.column("active_elo").to_numpy()
            tcs = batch.column("tc_cat").to_numpy()
            in_rng = batch.column("in_balance_range").to_numpy(zero_copy_only=False).astype(bool)
            oor += int((~in_rng).sum())
            in_elos = elos[in_rng]
            in_tcs = tcs[in_rng]
            bins = ((in_elos - ELO_MIN) // ((ELO_MAX - ELO_MIN) // NUM_ELO_BINS)).astype(np.int32)
            for b, t in zip(bins.tolist(), in_tcs.tolist()):
                counts[(int(b), int(t))] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(shards)} elapsed={time.time()-t0:.0f}s", flush=True)

    issues = []
    summary = {
        "oor_count": oor,
        "buckets": {},
        "extra_tc_counts": {TC_NAMES[t]: {} for t in EXTRA_TC_CATS},
        "extra_tc_targets": extra_tc_targets(target_per_bin),
    }
    for b in range(NUM_ELO_BINS):
        for t in range(NUM_TC_CATS):
            n = counts.get((b, t), 0)
            if t in EXTRA_TC_CATS:
                # Extra TCs are informational only — global caps are enforced during extraction.
                summary["extra_tc_counts"][TC_NAMES[t]][f"{b}"] = n
                continue
            tgt = targets.get(t, 0)
            summary["buckets"][f"{b}_{t}"] = {"count": n, "target": tgt}
            if n < int(tgt * 0.99):
                issues.append(f"bin{b} tc{t}: {n:,} < 99% of target {tgt:,}")
    if oor == 0:
        issues.append("no out-of-range rows present (in_balance_range=False)")
    summary["issues"] = issues
    return not issues, summary


def check_tc_consistency(shards: list, sample_frac: float = 0.01) -> Tuple[bool, list]:
    n = max(1, int(len(shards) * sample_frac))
    sample = random.sample(shards, min(n, len(shards)))
    issues = []
    for s in sample:
        t = pq.read_table(s, columns=["base_time", "active_inc", "tc_cat"])
        bases = t.column("base_time").to_numpy()
        incs = t.column("active_inc").to_numpy()
        cats = t.column("tc_cat").to_numpy()
        recomp = np.array([get_tc_category(int(b), int(i)) for b, i in zip(bases, incs)])
        mismatch = int(np.sum(recomp != cats))
        if mismatch:
            issues.append(f"{s.name}: {mismatch}/{len(cats)} tc_cat mismatch vs base+inc recompute")
    return not issues, issues


def check_oor_sanity(shards: list) -> Tuple[bool, list]:
    """For OOR rows, active_elo must lie outside [1200, 2500)."""
    sample = random.sample(shards, min(3, len(shards)))
    issues = []
    for s in sample:
        t = pq.read_table(s, columns=["active_elo", "in_balance_range"])
        elos = t.column("active_elo").to_numpy()
        in_rng = t.column("in_balance_range").to_numpy(zero_copy_only=False).astype(bool)
        bad = (in_rng & ((elos < ELO_MIN) | (elos >= ELO_MAX))).sum()
        bad += ((~in_rng) & (elos >= ELO_MIN) & (elos < ELO_MAX)).sum()
        if bad:
            issues.append(f"{s.name}: {int(bad)} rows with mismatched in_balance_range vs active_elo")
    return not issues, issues


def check_loadable(shards: list) -> Tuple[bool, list]:
    """Read 3 random shards with pyarrow and decode a few rows. Light-weight."""
    sample = random.sample(shards, min(3, len(shards)))
    issues = []
    for s in sample:
        try:
            t = pq.read_table(s)
            if t.num_rows < 1:
                issues.append(f"{s.name}: empty")
            # touch a few row dicts
            for col in ["board_history", "legal_moves", "policy_move"]:
                t.column(col).slice(0, 1).to_pylist()
        except Exception as e:
            issues.append(f"{s.name}: load failed: {e}")
    return not issues, issues


def check_nested_lengths(shards: list, sample_frac: float = 0.01) -> Tuple[bool, list]:
    """Sample shards and validate nested tensor lengths expected by dataset.py."""
    n = max(1, int(len(shards) * sample_frac))
    sample = random.sample(shards, min(n, len(shards)))
    issues = []
    for s in sample:
        try:
            pf = pq.ParquetFile(s)
            bad_board = 0
            bad_time = 0
            bad_rep = 0
            checked = 0
            for batch in pf.iter_batches(
                batch_size=65536,
                columns=["board_history", "time_history", "repetition_flags"],
            ):
                checked += batch.num_rows
                bad_board += sum(
                    1 for bh in batch["board_history"].to_pylist()
                    if len(bh) != 8 or any(len(pos) != 64 for pos in bh)
                )
                bad_time += sum(1 for th in batch["time_history"].to_pylist() if len(th) != 8)
                bad_rep += sum(1 for rf in batch["repetition_flags"].to_pylist() if len(rf) != 8)
            if bad_board or bad_time or bad_rep:
                issues.append(
                    f"{s.name}: bad board_history={bad_board}, time_history={bad_time}, repetition_flags={bad_rep} over {checked:,} rows"
                )
        except Exception as e:
            issues.append(f"{s.name}: nested-length scan failed: {e}")
    return not issues, issues


def run_verify(target_per_bin: int = DEFAULT_TARGET_PER_BIN) -> bool:
    shards = _verified_shards(TRAIN_DIR)
    print(f"[verify] {len(shards)} shards under {TRAIN_DIR}")
    if not shards:
        print("[verify] FAIL: no shards")
        return False

    report: dict = {"shards": len(shards), "checks": {}}

    ok_schema, schema_issues = check_schema(shards)
    report["checks"]["schema"] = {"ok": ok_schema, "issues": schema_issues}
    print(f"  schema:        {'OK' if ok_schema else 'FAIL'}  issues={len(schema_issues)}")

    ok_counts, counts_summary = check_counts(shards, target_per_bin)
    report["checks"]["counts"] = {"ok": ok_counts, "summary": counts_summary}
    print(f"  counts:        {'OK' if ok_counts else 'FAIL'}  "
          f"oor={counts_summary['oor_count']:,} issues={len(counts_summary['issues'])}")

    ok_tc, tc_issues = check_tc_consistency(shards)
    report["checks"]["tc_consistency"] = {"ok": ok_tc, "issues": tc_issues}
    print(f"  tc_consistency:{'OK' if ok_tc else 'FAIL'}  issues={len(tc_issues)}")

    ok_oor, oor_issues = check_oor_sanity(shards)
    report["checks"]["oor_sanity"] = {"ok": ok_oor, "issues": oor_issues}
    print(f"  oor_sanity:    {'OK' if ok_oor else 'FAIL'}  issues={len(oor_issues)}")

    ok_load, load_issues = check_loadable(shards)
    report["checks"]["loadable"] = {"ok": ok_load, "issues": load_issues}
    print(f"  loadable:      {'OK' if ok_load else 'FAIL'}  issues={len(load_issues)}")

    ok_nested, nested_issues = check_nested_lengths(shards)
    report["checks"]["nested_lengths"] = {"ok": ok_nested, "issues": nested_issues}
    print(f"  nested_lengths:{'OK' if ok_nested else 'FAIL'}  issues={len(nested_issues)}")

    overall = all([ok_schema, ok_counts, ok_tc, ok_oor, ok_load, ok_nested])
    report["overall_ok"] = overall

    VERIFY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(VERIFY_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[verify] {'PASS' if overall else 'FAIL'}. report: {VERIFY_REPORT}")
    return overall
