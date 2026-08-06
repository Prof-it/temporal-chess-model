"""CLI orchestrator for build_dataset.

Subcommands:
    default action: process pending months newest-first from --end backward
                    until the balanced core buckets are filled, then verify.
  --from-scratch: archive train/, clear state, exit.
  --balance-only: rerun phase E from existing stage/.
  --verify-only:  rerun phase F from existing train/.
  --status:       print state summary.
  --restart-month YYYY-MM: wipe one month's stage and reprocess.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

from . import balance as balance_mod
from . import http_zstd
from . import pgn_stream
from . import shuffle as shuffle_mod
from . import state as state_mod
from . import verify as verify_mod
from .config import (
    BALANCED_DIR,
    BALANCED_TC_CATS,
    DEFAULT_SEED,
    DEFAULT_TARGET_PER_BIN,
    DEFAULT_WORKERS,
    ELO_MAX,
    ELO_MIN,
    ELO_STEP,
    EXTRA_TC_CATS,
    MANIFEST_FILE,
    NUM_ELO_BINS,
    STAGE_DIR,
    STATE_FILE,
    TC_NAMES,
    TC_THRESHOLDS,
    TRAIN_DIR,
    ULTRABULLET_MAX,
    elo_bin,
    extra_tc_targets,
    oor_target,
    target_per_bucket,
)
from .extractor import init_worker, process_game_text
from .staging import MonthlyWriter


# ─── Month range helpers ─────────────────────────────────────────────────────
def parse_month(s: str) -> dt.date:
    y, m = s.split("-")
    return dt.date(int(y), int(m), 1)


def fmt_month(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_range(start: str, end: str, newest_first: bool = True) -> List[str]:
    """Inclusive list of YYYY-MM strings from start to end.

    By default the builder works newest-first so recent data fills the core
    balanced buckets before we scan older months.
    """
    cur = parse_month(start)
    stop = parse_month(end)
    out = []
    while cur <= stop:
        out.append(fmt_month(cur))
        # advance one month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)
    if newest_first:
        out.reverse()
    return out


def default_end() -> str:
    """Default end month is the previous full month (current month often unpublished)."""
    today = dt.date.today()
    if today.month == 1:
        return f"{today.year - 1:04d}-12"
    return f"{today.year:04d}-{today.month - 1:02d}"


def default_oldest_month() -> str:
    """Safety floor for backward scans when no oldest month is specified."""
    return "2013-01"


# ─── Per-month processing ────────────────────────────────────────────────────
_INF_CAP = 10**12


def build_adaptive_caps(
    target_per_bin: int,
    num_months_remaining: int,
    cumulative_kept: Dict[Tuple[int, int], int],
    num_workers: int,
    is_last_month: bool,
    tc_filter: Optional[int] = None,
) -> Dict[Tuple[int, int], int]:
    """Per-worker per-bucket cap for the next month, derived from running deficit.

    Caps are built only for balanced TC categories. If tc_filter is set, only
    that single TC gets caps registered (classical and other TCs filtered out
    upstream by the extractor).

    In the recent-first builder we want each month to fill as much of the
    remaining deficit as possible before moving backward, so the per-worker cap
    is based on the full remaining deficit rather than spreading it across the
    whole historical window.
    """
    targets = target_per_bucket(target_per_bin)
    tcs = (tc_filter,) if tc_filter is not None else BALANCED_TC_CATS
    caps: Dict[Tuple[int, int], int] = {}
    for b in range(NUM_ELO_BINS):
        for t in tcs:
            target_for_bucket = targets[t]
            kept = cumulative_kept.get((b, t), 0)
            deficit = max(0, target_for_bucket - kept)
            if deficit == 0:
                caps[(b, t)] = 1
            else:
                per_worker = max(1, int(deficit / max(1, num_workers)))
                caps[(b, t)] = per_worker
    return caps


def aggregate_yields_to_bins(month_yields: dict) -> Dict[Tuple[int, int], int]:
    """Roll per-(active_elo, tc_cat) yields up to per-(elo_bin, tc_cat).

    Out-of-range elos are dropped from this aggregate (they aren't subject to caps).
    """
    out: Dict[Tuple[int, int], int] = defaultdict(int)
    for key, cnt in month_yields.items():
        elo_str, tc_str = key.split("_")
        b = elo_bin(int(elo_str))
        if b < 0:
            continue
        out[(b, int(tc_str))] += int(cnt)
    return out


def aggregate_oor_yield(month_yields: dict) -> int:
    """Sum all OOR (active_elo outside [1200, 2500)) yields from a month."""
    total = 0
    for key, cnt in month_yields.items():
        elo_str, _ = key.split("_")
        if elo_bin(int(elo_str)) < 0:
            total += int(cnt)
    return total


def aggregate_tc_yield(month_yields: dict, tc_cat: int) -> int:
    """Sum all rows for one tc_cat across all Elo values."""
    total = 0
    for key, cnt in month_yields.items():
        _, tc_str = key.split("_")
        if int(tc_str) == tc_cat:
            total += int(cnt)
    return total


def build_adaptive_oor_cap(
    target_per_bin: int,
    num_months_remaining: int,
    cumulative_oor: int,
    num_workers: int,
    is_last_month: bool,
    tc_filter: Optional[int] = None,
) -> int:
    """Per-worker OOR cap for the next month. 0 means uncapped.

    With tc_filter, OOR target is scaled to a single-TC fraction (since only
    that TC's games contribute OOR rows in a staged build).
    """
    target = oor_target(target_per_bin)
    if tc_filter is not None:
        # Single-TC stage: OOR target scaled to 1/3 of the full-balance OOR.
        target = target // 3
    if target <= 0:
        return 0
    deficit = max(0, target - cumulative_oor)
    if deficit == 0:
        return 1
    return max(1, int(deficit / max(1, num_workers)))


def build_adaptive_extra_tc_caps(
    target_per_bin: int,
    num_months_remaining: int,
    cumulative_extra_tcs: Dict[int, int],
    num_workers: int,
    is_last_month: bool,
    tc_filter: Optional[int] = None,
) -> Dict[int, int]:
    """Per-worker caps for low-priority extra TCs with small global budgets."""
    if tc_filter is not None:
        return {}
    caps: Dict[int, int] = {}
    for tc_cat, target in extra_tc_targets(target_per_bin).items():
        kept = cumulative_extra_tcs.get(tc_cat, 0)
        deficit = max(0, target - kept)
        if deficit == 0:
            caps[tc_cat] = 1
        else:
            caps[tc_cat] = max(1, int(deficit / max(1, num_workers)))
    return caps


def _local_byte_iter(path: str, chunk_size: int = 1 << 20):
    """Stream a local file as raw bytes for testing without HTTP/zstd."""
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                return
            yield b


def _all_balanced_caps_met(
    writer_yields: Dict[Tuple[int, int], int],
    per_worker_caps: Dict[Tuple[int, int], int],
    num_workers: int,
) -> bool:
    """Check if all balanced (bin × tc) buckets reached their per-month aggregate cap.

    The aggregate cap is per_worker_cap × num_workers (caps in the dict are PER WORKER).
    Yields are keyed by active_elo at row write time; we re-bin them here.
    """
    bin_yields = aggregate_yields_to_bins(
        {f"{elo}_{tc}": cnt for (elo, tc), cnt in writer_yields.items()}
    )
    for (b, t), per_worker_cap in per_worker_caps.items():
        if t not in BALANCED_TC_CATS:
            continue
        agg_cap = per_worker_cap * num_workers
        if bin_yields.get((b, t), 0) < agg_cap:
            return False
    return True


def _all_global_targets_met(
    cumulative_kept: Dict[Tuple[int, int], int],
    target_per_bin: int,
    tc_filter: Optional[int] = None,
) -> bool:
    """Check whether the global balanced objective has been met."""
    targets = target_per_bucket(target_per_bin)
    tcs = (tc_filter,) if tc_filter is not None else BALANCED_TC_CATS
    for b in range(NUM_ELO_BINS):
        for t in tcs:
            if cumulative_kept.get((b, t), 0) < targets[t]:
                return False
    return True


def process_month(
    month: str,
    workers: int,
    seed: int,
    per_worker_caps: Dict[Tuple[int, int], int],
    extra_tc_caps_per_worker: Dict[int, int],
    local_pgn: Optional[str] = None,
    max_games: int = 0,
    max_minutes: float = 0,
    oor_cap_per_worker: int = 0,
    tc_filter: Optional[int] = None,
) -> Tuple[int, int, dict]:
    """Stream + extract one month. Returns (games_seen, rows_kept, yields_summary).

    If local_pgn is given, reads from that file instead of downloading.
    Stops early when:
      - max_games reached (testing), or
      - all balanced caps for this month met (full-run optimization), or
      - max_minutes elapsed (per-month time budget; deficit propagates to next month).
    """
    print(f"\n[{month}] starting", flush=True)
    t0 = time.time()
    writer = MonthlyWriter(month, seed=seed)

    bytes_total = [0]
    last_print = [time.time()]

    def progress(n_bytes: int) -> None:
        bytes_total[0] += n_bytes
        # Throttle prints to once per 5s
        now = time.time()
        if now - last_print[0] >= 5.0:
            mb = bytes_total[0] / 1e6
            print(f"  [{month}] decompressed: {mb:,.0f} MB, "
                  f"games_seen={writer.games_seen:,}, rows_kept={writer.rows_written + len(writer.buffer):,}",
                  flush=True)
            last_print[0] = now

    try:
        if local_pgn:
            print(f"  [{month}] using local file: {local_pgn}")
            byte_iter = _local_byte_iter(local_pgn)
        else:
            byte_iter = http_zstd.stream_pgn_bytes(month, progress_cb=progress)
        game_iter = pgn_stream.iter_game_blocks(byte_iter)

        with Pool(
            workers,
            initializer=init_worker,
            initargs=(0, 0, seed, per_worker_caps, extra_tc_caps_per_worker,
                      oor_cap_per_worker, tc_filter),
        ) as pool:
            # We pass worker_id=0 for simplicity; per-worker rng differentiation
            # uses Pool's PID. The per-worker cap is total/num_workers so collective
            # yield ≈ aggregate cap regardless of which worker eats which game.
            count = 0
            CHECK_EVERY = 5000  # check termination conditions every N games
            stop_reason: Optional[str] = None
            for rows in pool.imap_unordered(process_game_text, game_iter, chunksize=16):
                count += 1
                writer.add_games_seen(1)
                if rows:
                    writer.add_rows(rows)
                if max_games and count >= max_games:
                    stop_reason = f"max-games={max_games}"
                elif count % CHECK_EVERY == 0:
                    if max_minutes > 0 and (time.time() - t0) >= max_minutes * 60:
                        stop_reason = f"max-minutes={max_minutes}"
                    elif _all_balanced_caps_met(writer.yields, per_worker_caps, workers):
                        stop_reason = "all balanced caps met"
                if stop_reason:
                    print(f"  [{month}] stopping early: {stop_reason}", flush=True)
                    pool.terminate()
                    break

        writer.finalize(status="extracted")
    except http_zstd.MonthUnavailable:
        writer.finalize(status="failed", error="month_not_published")
        # Wipe the .tmp dir; nothing useful there.
        shutil.rmtree(STAGE_DIR / f"{month}.tmp", ignore_errors=True)
        return 0, 0, {"status": "unavailable"}
    except Exception as e:
        writer.finalize(status="failed", error=str(e))
        raise

    elapsed = time.time() - t0
    yields_compact = {
        f"{elo}_{tc}": cnt for (elo, tc), cnt in writer.yields.items()
    }
    print(f"[{month}] done. games={writer.games_seen:,} rows={writer.rows_written:,} "
          f"shards={writer.shards_written} elapsed={elapsed/60:.1f}m", flush=True)
    return writer.games_seen, writer.rows_written, {
        "status": "extracted",
        "yields": yields_compact,
        "elapsed_seconds": int(elapsed),
    }


# ─── Top-level subcommands ───────────────────────────────────────────────────
def cmd_status(state: dict) -> int:
    months = state.get("months", {})
    by_status: Dict[str, int] = {}
    total_games = total_rows = 0
    for m in months.values():
        by_status[m["status"]] = by_status.get(m["status"], 0) + 1
        total_games += m.get("games_seen", 0) or 0
        total_rows += m.get("rows_kept", 0) or 0
    print(f"build_state: {STATE_FILE}")
    print(f"  config: {state.get('config')}")
    print(f"  months: {len(months)}")
    for s, n in sorted(by_status.items()):
        print(f"    {s:>13s}: {n}")
    print(f"  total games_seen: {total_games:,}")
    print(f"  total rows_kept:  {total_rows:,}")
    print(f"  balance_pass:     {state.get('balance_pass', {}).get('status')}")
    return 0


def cmd_from_scratch() -> int:
    if TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir()):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = BALANCED_DIR / f"train.archived.{ts}"
        print(f"archiving {TRAIN_DIR} → {archive}")
        shutil.move(str(TRAIN_DIR), str(archive))
    if STAGE_DIR.exists():
        print(f"removing {STAGE_DIR}")
        shutil.rmtree(STAGE_DIR)
    for p in [STATE_FILE, MANIFEST_FILE]:
        if p.exists():
            print(f"removing {p}")
            p.unlink()
    print("cleared. rerun `python -m build_dataset` without --from-scratch to start.")
    return 0


def cmd_restart_month(state: dict, month: str) -> int:
    state_mod.reset_month(state, month)
    state_mod.save_state(state)
    # Wipe both the committed direct-to-final month dir and any .tmp leftover.
    for p in [TRAIN_DIR / month, TRAIN_DIR / f"{month}.tmp"]:
        if p.exists():
            print(f"removing {p}")
            shutil.rmtree(p)
    print(f"month {month} reset; will be reprocessed on next run.")
    return 0


def cmd_balance_only(target_per_bin: int, seed: int) -> int:
    """LEGACY — only valid for old 2-pass datasets that still have data/train/stage/."""
    print("[balance-only] LEGACY: this requires data/train/stage/ to exist.")
    print("[balance-only] The default pipeline now writes directly to train/YYYY-MM/ — no balance pass needed.")
    if not STAGE_DIR.exists():
        print(f"[balance-only] no stage dir at {STAGE_DIR} — nothing to do.")
        return 1
    summary = balance_mod.run_balance_pass(target_per_bin=target_per_bin, seed=seed)
    state = state_mod.load_state()
    state["balance_pass"] = {
        "status": "done",
        "started_at": state_mod.now_str(),
        "finished_at": state_mod.now_str(),
        "summary": summary,
    }
    state_mod.save_state(state)
    print("[balance-only] done (legacy 2-pass)")
    return 0


def cmd_verify_only() -> int:
    return 0 if verify_mod.run_verify() else 1


def cmd_reshuffle(seed: int, delete_inputs: bool, num_partitions: Optional[int]) -> int:
    """Global shuffle of all per-month shards into train/shuffled/."""
    print("[reshuffle] starting global shuffle pass over train/")
    summary = shuffle_mod.run_shuffle(
        seed=seed,
        delete_inputs=delete_inputs,
        num_partitions=num_partitions,
    )
    print()
    print(f"[reshuffle] summary: {summary}")
    return 0


def write_manifest(state: dict, summary: dict, target_per_bin: int) -> None:
    manifest = {
        "version": 4,
        "created": state_mod.now_str(),
        "design": "direct-to-final with adaptive caps, newest-first month order, and stop-when-filled core buckets",
        "schema": {
            "tc_cat_encoding": {name: i for i, name in enumerate(TC_NAMES)},
            "tc_thresholds_seconds": {
                "ultraBullet": ULTRABULLET_MAX,
                "bullet": TC_THRESHOLDS[0],
                "blitz": TC_THRESHOLDS[1],
                "rapid": TC_THRESHOLDS[2],
            },
            "elo_bin_min": ELO_MIN,
            "elo_bin_max": ELO_MAX,
            "elo_bin_step": ELO_STEP,
            "in_balance_range": "true iff 1200 <= active_elo < 2500",
        },
        "config": state.get("config"),
        "months_processed": [m for m, info in state["months"].items()
                             if info["status"] == "extracted"],
        "per_month_yields": {m: info.get("yields", {})
                              for m, info in state["months"].items()
                              if info["status"] == "extracted"},
        "target_per_bin": target_per_bin,
        "target_per_bucket": target_per_bucket(target_per_bin),
        "extra_tc_targets": extra_tc_targets(target_per_bin),
        "summary": summary,
    }
    BALANCED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def cmd_run(args: argparse.Namespace) -> int:
    months = month_range(args.oldest_month, args.end, newest_first=True)
    print(
        f"month range: {args.oldest_month} → {args.end} "
        f"({len(months)} months, newest-first)"
    )

    state = state_mod.load_state()
    if state["config"] is None:
        state["config"] = {
            "month_range": [args.oldest_month, args.end],
            "month_order": "newest-first",
            "newest_month": args.end,
            "oldest_month": args.oldest_month,
            "target_per_bin": args.target_per_bin,
            "tc_targets": target_per_bucket(args.target_per_bin),
            "extra_tc_targets": extra_tc_targets(args.target_per_bin),
            "seed": args.seed,
            "workers": args.workers,
        }
        state_mod.save_state(state)

    # Rebuild cumulative_kept and cumulative_oor from already-extracted months
    # so resume works correctly.
    cumulative_kept: Dict[Tuple[int, int], int] = defaultdict(int)
    cumulative_oor = 0
    cumulative_extra_tcs: Dict[int, int] = defaultdict(int)
    for m in months:
        info = state["months"].get(m, {})
        if info.get("status") == "extracted":
            for k, v in aggregate_yields_to_bins(info.get("yields", {})).items():
                cumulative_kept[k] += v
            cumulative_oor += aggregate_oor_yield(info.get("yields", {}))
            for tc_cat in EXTRA_TC_CATS:
                cumulative_extra_tcs[tc_cat] += aggregate_tc_yield(info.get("yields", {}), tc_cat)

    pending = [m for m in months if not state_mod.month_is_complete(state, m)]
    print(f"pending: {len(pending)} (already extracted: {len(months) - len(pending)})")
    if args.dry_run:
        print("dry-run: would process pending months, no balance pass needed.")
        return 0

    targets = target_per_bucket(args.target_per_bin)
    print(f"per-bucket targets: {targets}")
    print(f"extra tc targets: {extra_tc_targets(args.target_per_bin)}")

    if _all_global_targets_met(cumulative_kept, args.target_per_bin, args.tc_filter_cat):
        print("core balanced targets already met; skipping additional months.")
        pending = []

    for i, month in enumerate(pending):
        months_remaining = len(pending) - i
        is_last = (i == len(pending) - 1)
        per_worker_caps = build_adaptive_caps(
            target_per_bin=args.target_per_bin,
            num_months_remaining=months_remaining,
            cumulative_kept=cumulative_kept,
            num_workers=args.workers,
            is_last_month=is_last,
            tc_filter=args.tc_filter_cat,
        )
        extra_tc_caps = build_adaptive_extra_tc_caps(
            target_per_bin=args.target_per_bin,
            num_months_remaining=months_remaining,
            cumulative_extra_tcs=cumulative_extra_tcs,
            num_workers=args.workers,
            is_last_month=is_last,
            tc_filter=args.tc_filter_cat,
        )
        oor_cap = build_adaptive_oor_cap(
            target_per_bin=args.target_per_bin,
            num_months_remaining=months_remaining,
            cumulative_oor=cumulative_oor,
            num_workers=args.workers,
            is_last_month=is_last,
            tc_filter=args.tc_filter_cat,
        )
        # Diagnostic: show a sample cap so user can see them adapt.
        # Picks one bucket from the filtered TC, or one per balanced TC.
        if args.tc_filter_cat is not None:
            t = args.tc_filter_cat
            sample_keys = [(0, t), (6, t), (12, t)]
        else:
            sample_keys = [(0, 0), (6, 1), (12, 2)]
        sample_caps = {f"bin{b}_tc{t}": per_worker_caps[(b, t)]
                       for b, t in sample_keys}
        extra_caps_named = {TC_NAMES[tc]: cap for tc, cap in extra_tc_caps.items()}
        print(f"[{month}] adaptive caps (per-worker, sample): {sample_caps}, "
              f"extra_tc_caps={extra_caps_named}, oor_cap={oor_cap:,}, "
              f"months_remaining={months_remaining}, last={is_last}")

        state_mod.set_month_status(state, month, "downloading",
                                   url=http_zstd.url_for(month),
                                   started_at=state_mod.now_str())
        state_mod.save_state(state)
        try:
            games, rows, info = process_month(month, args.workers, args.seed, per_worker_caps,
                                              extra_tc_caps,
                                              local_pgn=args.local_pgn,
                                              max_games=args.max_games,
                                              max_minutes=args.max_minutes_per_month,
                                              oor_cap_per_worker=oor_cap,
                                              tc_filter=args.tc_filter_cat)
        except Exception as e:
            state_mod.set_month_status(state, month, "failed",
                                       error=str(e),
                                       finished_at=state_mod.now_str())
            state_mod.save_state(state)
            print(f"[{month}] FAILED: {e}", file=sys.stderr)
            if args.stop_on_error:
                return 2
            continue
        if info["status"] == "unavailable":
            state_mod.set_month_status(state, month, "unavailable",
                                       finished_at=state_mod.now_str())
        else:
            state_mod.set_month_status(state, month, "extracted",
                                       games_seen=games, rows_kept=rows,
                                       yields=info["yields"],
                                       finished_at=state_mod.now_str(),
                                       elapsed_seconds=info["elapsed_seconds"])
            # Update running cumulative for next month's cap calc.
            for k, v in aggregate_yields_to_bins(info["yields"]).items():
                cumulative_kept[k] += v
            cumulative_oor += aggregate_oor_yield(info["yields"])
            for tc_cat in EXTRA_TC_CATS:
                cumulative_extra_tcs[tc_cat] += aggregate_tc_yield(info["yields"], tc_cat)
        state_mod.save_state(state)
        if _all_global_targets_met(cumulative_kept, args.target_per_bin, args.tc_filter_cat):
            print(f"[{month}] core balanced targets filled globally; stopping.", flush=True)
            break

    summary = build_summary(state, target_per_bin=args.target_per_bin,
                            cumulative_kept=cumulative_kept)
    write_manifest(state, summary, target_per_bin=args.target_per_bin)
    print("\n[verify] running verification…")
    ok = verify_mod.run_verify(target_per_bin=args.target_per_bin)
    return 0 if ok else 1


def build_summary(state: dict, target_per_bin: int,
                  cumulative_kept: Dict[Tuple[int, int], int]) -> dict:
    """Final per-bucket counts assembled from per-month yields (no balance pass).

    Classical and ultraBullet are reported separately as informational capped
    extras; only bullet, blitz, and rapid are subject to hard balance targets.
    """
    targets = target_per_bucket(target_per_bin)
    final_balanced = {
        f"{b}_{t}": cumulative_kept.get((b, t), 0)
        for b in range(NUM_ELO_BINS)
        for t in BALANCED_TC_CATS
    }
    final_extra_per_bin = {
        TC_NAMES[t]: {
            f"{b}": cumulative_kept.get((b, t), 0)
            for b in range(NUM_ELO_BINS)
        }
        for t in EXTRA_TC_CATS
    }
    # OOR = sum across months (active_elo outside [1200,2500))
    oor = 0
    extra_totals = {TC_NAMES[t]: 0 for t in EXTRA_TC_CATS}
    for info in state["months"].values():
        if info.get("status") != "extracted":
            continue
        for tc_cat in EXTRA_TC_CATS:
            extra_totals[TC_NAMES[tc_cat]] += aggregate_tc_yield(info.get("yields", {}), tc_cat)
        for key, cnt in info.get("yields", {}).items():
            elo_str, _ = key.split("_")
            if elo_bin(int(elo_str)) < 0:
                oor += int(cnt)
    return {
        "final_balanced": final_balanced,
        "final_extra_per_bin": final_extra_per_bin,
        "final_extra_totals": extra_totals,
        "final_out_of_range": oor,
        "total_rows": sum(
            info.get("rows_kept", 0)
            for info in state["months"].values()
            if info.get("status") == "extracted"
        ),
        "targets": targets,
        "extra_tc_targets": extra_tc_targets(target_per_bin),
        "deficits": {
            f"{b}_{t}": max(0, targets[t] - cumulative_kept.get((b, t), 0))
            for b in range(NUM_ELO_BINS)
            for t in BALANCED_TC_CATS
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(prog="build_dataset")
    p.add_argument("--end", default=None,
                   help="Newest month YYYY-MM to start from (default: previous full month)")
    p.add_argument("--oldest-month", default=None,
                   help="Optional oldest month YYYY-MM safety floor. The build stops earlier if the core buckets fill first.")
    p.add_argument("--start", dest="deprecated_start", default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--target-per-bin", type=int, default=DEFAULT_TARGET_PER_BIN)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, no I/O")
    p.add_argument("--from-scratch", action="store_true",
                   help="Archive train/, clear state, exit. Rerun without flag to start.")
    p.add_argument("--balance-only", action="store_true",
                   help="Run only the balance pass on existing stage/.")
    p.add_argument("--verify-only", action="store_true",
                   help="Run only the verify pass on existing train/.")
    p.add_argument("--status", action="store_true",
                   help="Print state summary and exit.")
    p.add_argument("--restart-month", type=str, default=None,
                   help="Wipe and reprocess one month YYYY-MM.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort on first month failure (default: skip and continue).")
    p.add_argument("--local-pgn", type=str, default=None,
                   help="Process this local .pgn file instead of downloading from Lichess. "
                        "Used with --end and optionally --oldest-month specifying a single month label for staging.")
    p.add_argument("--max-games", type=int, default=0,
                   help="Stop after processing N games per month (testing). 0 = no limit.")
    p.add_argument("--tc-filter", type=str, default=None,
                   choices=["bullet", "blitz", "rapid"],
                   help="Process only games of this time-control category. Used for staged "
                        "single-TC builds: run bullet stage to one MARVIN_DATA_DIR, move it, "
                        "then run blitz stage to a fresh dir, etc. Other TCs (including "
                        "classical and ultraBullet) are skipped at header-parse time. Target per (bin × TC) "
                        "is still target_per_bin / 3, so each stage hits the same bucket size "
                        "as a full run would for that TC.")
    p.add_argument("--max-minutes-per-month", type=float, default=0.0,
                   help="Per-month wall-clock budget (safety valve). After this many minutes, "
                        "finalize the month with whatever was kept and move on (deficit propagates "
                        "to next month's adaptive cap). 0 = no limit (default — pre-skip keeps "
                        "per-month time bounded by HTTP throughput instead).")
    p.add_argument("--reshuffle", action="store_true",
                   help="After the build completes, run a global cross-shard shuffle pass. "
                        "Reads train/<month>/*.parquet, writes train/shuffled/*.parquet with "
                        "uniform mixed distribution.")
    p.add_argument("--shuffle-only", action="store_true",
                   help="Run only the shuffle pass on existing train/<month>/*.parquet. "
                        "Skips download/extract.")
    p.add_argument("--reshuffle-delete-inputs", action="store_true",
                   help="With --reshuffle: delete per-month input dirs after shuffled output is "
                        "written. Saves ~85 GB. Irreversible.")
    p.add_argument("--reshuffle-partitions", type=int, default=None,
                   help="K partitions for the shuffle pass. Default: auto (16-256 based on "
                        "input size).")
    args = p.parse_args()
    if args.end is None:
        args.end = default_end()
    if args.oldest_month is None:
        args.oldest_month = args.deprecated_start or default_oldest_month()
    elif args.deprecated_start is not None:
        print("[build_dataset] ignoring deprecated --start because --oldest-month was provided.", flush=True)

    if parse_month(args.oldest_month) > parse_month(args.end):
        p.error("--oldest-month must be earlier than or equal to --end")

    # Resolve --tc-filter name → tc_cat int (or None).
    tc_name_to_cat = {name: i for i, name in enumerate(TC_NAMES)}
    args.tc_filter_cat = tc_name_to_cat.get(args.tc_filter) if args.tc_filter else None

    if args.from_scratch:
        return cmd_from_scratch()
    if args.status:
        return cmd_status(state_mod.load_state())
    if args.restart_month:
        return cmd_restart_month(state_mod.load_state(), args.restart_month)
    if args.balance_only:
        return cmd_balance_only(args.target_per_bin, args.seed)
    if args.verify_only:
        return cmd_verify_only()
    if args.shuffle_only:
        return cmd_reshuffle(args.seed, args.reshuffle_delete_inputs, args.reshuffle_partitions)
    rc = cmd_run(args)
    if rc == 0 and args.reshuffle:
        cmd_reshuffle(args.seed, args.reshuffle_delete_inputs, args.reshuffle_partitions)
    return rc


if __name__ == "__main__":
    sys.exit(main())
