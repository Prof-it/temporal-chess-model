"""Seed-sweep evaluation: paired cond-vs-blind deltas across three seed pairs.

Runs the six 250k-step checkpoints (three time-conditioned, three time-blind)
over one shared eval set in a single pass, so every per-row metric array is
row-aligned across all six models. That alignment is what makes the paired
delta (conditioned minus blind, *same seed pair*) meaningful: each delta is
computed on identical positions, so position difficulty cancels out.

Reuses :class:`evaluation.metrics.MetricsAccumulator` unchanged — the seed
sweep must be comparable to the existing single-seed headline numbers, so it
has to be the same harness, not a reimplementation.

With n=3 seed pairs this deliberately reports **mean and range**, never a CI or
an SD-based interval: three points do not support a distributional claim.

Outputs a JSON summary plus an .npz of the raw per-row arrays so the analysis
can be redone without re-running the (expensive) forward passes.

Example:
    python -m evaluation.seed_sweep \
        --pairs s67=ckpt/marvin_23m_cond_250k_s67.pt,ckpt/marvin_23m_blind_250k_s67.pt \
                s68=ckpt/marvin_23m_cond_250k_s68.pt,ckpt/marvin_23m_blind_250k_s68.pt \
                s69=ckpt/marvin_23m_cond_250k_s69.pt,ckpt/marvin_23m_blind_250k_s69.pt \
        --eval-set data/testset_2026-05 \
        --out eval_results/paper_2026/seed_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backends import make_backend
from .data import iter_eval_batches, resolve_eval_files, BUCKET_NAMES
from .metrics import MetricsAccumulator

CRITICAL = 3   # index of the "Critical" (<5% clock remaining) bucket


def parse_pairs(specs: list[str]) -> list[tuple[str, Path, Path]]:
    """Parse ``label=cond.pt,blind.pt`` specs into (label, cond, blind) triples."""
    pairs: list[tuple[str, Path, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--pairs entry must be label=cond.pt,blind.pt — got {spec!r}")
        label, paths = spec.split("=", 1)
        if paths.count(",") != 1:
            raise SystemExit(f"--pairs entry needs exactly two comma-separated "
                             f"checkpoints — got {spec!r}")
        cond, blind = (p.strip() for p in paths.split(","))
        pairs.append((label.strip(), Path(cond), Path(blind)))
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed-sweep paired evaluation")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="One label=cond.pt,blind.pt spec per matched seed pair. "
                        "The paper uses three pairs of 250k-step checkpoints "
                        "(seeds 67/68/69); pair s67 is the 250k archive of the "
                        "two 1M-step headline runs.")
    p.add_argument("--eval-set", default="data/testset_2026-05")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap rows (smoke tests only — the paper number needs the full set)")
    p.add_argument("--out", default="eval_results/paper_2026/seed_sweep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files, set_label = resolve_eval_files(args.eval_set)
    seed_pairs = parse_pairs(args.pairs)

    specs: list[tuple[str, Path]] = []
    for pair, cond, blind in seed_pairs:
        specs.append((f"cond_{pair}", cond))
        specs.append((f"blind_{pair}", blind))
    for label, path in specs:
        if not path.exists():
            raise SystemExit(f"missing checkpoint: {path}")

    print(f"[sweep] set={set_label} shards={len(files)} models={[s[0] for s in specs]}")
    backends = {label: make_backend(str(path), device=args.device, config_name=label)
                for label, path in specs}
    accums = {label: MetricsAccumulator(has_time_head=getattr(be, "has_time_head", True))
              for label, be in backends.items()}

    for batch in iter_eval_batches(files, batch_size=args.batch_size, limit=args.limit):
        for label, be in backends.items():
            accums[label].update(be(batch.inputs, return_promo=True), batch.metas)

    arrays = {label: acc._arrays() for label, acc in accums.items()}

    # Row alignment is the premise of every paired number below — verify it
    # rather than trusting it. All models see identical rows in identical order,
    # so the bucket vector (derived from row metadata, not model output) must be
    # byte-identical across models.
    ref_label = specs[0][0]
    ref_bucket = arrays[ref_label]["bucket"]
    for label, a in arrays.items():
        if a["bucket"].shape != ref_bucket.shape or not np.array_equal(a["bucket"], ref_bucket):
            raise SystemExit(f"row misalignment: {label} bucket vector differs from {ref_label}")
    n = ref_bucket.size
    print(f"[sweep] row-aligned across {len(arrays)} models, n={n:,}")

    def acc_of(label: str, mask: np.ndarray | None = None) -> float:
        mc = arrays[label]["move_correct"]
        return float(mc.mean() if mask is None else mc[mask].mean())

    def mae_of(label: str, mask: np.ndarray | None = None) -> float:
        te = arrays[label]["time_abs_err"]
        if te.size == 0:
            return float("nan")
        return float(te.mean() if mask is None else te[mask].mean())

    masks = {name: (ref_bucket == b) for b, name in enumerate(BUCKET_NAMES)}
    crit_mask = masks[BUCKET_NAMES[CRITICAL]]

    # ---- per-run levels -------------------------------------------------
    runs: dict[str, dict] = {}
    for label, _ in specs:
        runs[label] = {
            "n": int(n),
            "top1_overall": acc_of(label),
            "top1_by_bucket": {name: acc_of(label, m) for name, m in masks.items()},
            "wdl_accuracy": float(arrays[label]["wdl_correct"].mean()),
            "time_mae_overall_s": mae_of(label),
            "time_mae_critical_s": mae_of(label, crit_mask),
        }

    # ---- paired deltas, one per seed pair -------------------------------
    deltas: dict[str, dict] = {}
    for pair, _, _ in seed_pairs:
        c, b = f"cond_{pair}", f"blind_{pair}"
        deltas[pair] = {
            "overall": acc_of(c) - acc_of(b),
            "by_bucket": {name: acc_of(c, m) - acc_of(b, m) for name, m in masks.items()},
            "time_mae_overall_s": mae_of(c) - mae_of(b),
            "time_mae_critical_s": mae_of(c, crit_mask) - mae_of(b, crit_mask),
        }

    def mean_range(vals: list[float]) -> dict:
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals)), "range": float(np.max(vals) - np.min(vals)),
                "values": [float(v) for v in vals]}

    summary = {
        "overall": mean_range([deltas[p]["overall"] for p, _, _ in seed_pairs]),
        "by_bucket": {name: mean_range([deltas[p]["by_bucket"][name] for p, _, _ in seed_pairs])
                      for name in BUCKET_NAMES},
    }

    # ---- claim check 1: do the two arms' seed ranges overlap? -----------
    def arm_range(prefix: str, mask: np.ndarray | None) -> dict:
        vals = [acc_of(f"{prefix}_{p}", mask) for p, _, _ in seed_pairs]
        return {"min": float(np.min(vals)), "max": float(np.max(vals)),
                "values": [float(v) for v in vals]}

    overlap = {}
    for scope, mask in (("overall", None), (BUCKET_NAMES[CRITICAL], crit_mask)):
        cond_r, blind_r = arm_range("cond", mask), arm_range("blind", mask)
        overlap[scope] = {
            "cond": cond_r, "blind": blind_r,
            # Arms are separated iff the worst conditioned seed still beats the
            # best time-blind seed.
            "overlaps": bool(cond_r["min"] <= blind_r["max"]),
            "gap": float(cond_r["min"] - blind_r["max"]),
        }

    # ---- claim check 2: monotonic bucket ordering, per seed pair --------
    mono = {}
    for pair, _, _ in seed_pairs:
        seq = [deltas[pair]["by_bucket"][name] for name in BUCKET_NAMES]
        mono[pair] = {"deltas_in_bucket_order": [float(v) for v in seq],
                      "monotonic": bool(all(x < y for x, y in zip(seq, seq[1:])))}
    mono["holds_in_all_pairs"] = bool(all(mono[p]["monotonic"] for p, _, _ in seed_pairs))

    # ---- claim check 3: level spread vs delta spread --------------------
    # If absolute accuracy wobbles across seeds more than the paired delta does,
    # a single-seed headline table is defensible: the seed noise the reviewer
    # worries about largely cancels inside the pairing.
    stability = {}
    for scope, mask in (("overall", None), (BUCKET_NAMES[CRITICAL], crit_mask)):
        cond_vals = [acc_of(f"cond_{p}", mask) for p, _, _ in seed_pairs]
        blind_vals = [acc_of(f"blind_{p}", mask) for p, _, _ in seed_pairs]
        d_vals = [c - b for c, b in zip(cond_vals, blind_vals)]
        lvl = max(np.ptp(cond_vals), np.ptp(blind_vals))
        stability[scope] = {
            "level_range_cond": float(np.ptp(cond_vals)),
            "level_range_blind": float(np.ptp(blind_vals)),
            "delta_range": float(np.ptp(d_vals)),
            "level_over_delta": float(lvl / np.ptp(d_vals)) if np.ptp(d_vals) > 0 else float("inf"),
        }

    payload = {"eval_set": set_label, "n_shards": len(files), "n_positions": int(n),
               "checkpoints": {label: str(path) for label, path in specs},
               "runs": runs, "paired_deltas": deltas, "delta_summary": summary,
               "arm_overlap": overlap, "monotonicity": mono, "stability": stability}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"seed_sweep_{set_label}.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        out_dir / f"seed_sweep_arrays_{set_label}.npz",
        bucket=ref_bucket,
        **{f"{label}__{k}": v for label, a in arrays.items()
           for k, v in a.items() if k in ("move_correct", "wdl_correct", "time_abs_err")},
    )
    _print_report(payload)
    print(f"\n[sweep] wrote {out_dir}/seed_sweep_{set_label}.json")


def _print_report(d: dict) -> None:
    pairs = list(d["paired_deltas"])
    print("\n" + "=" * 78)
    print(f"SEED SWEEP — {d['eval_set']}  n={d['n_positions']:,} positions")
    print("=" * 78)

    print("\n--- Top-1 accuracy (%) ---")
    hdr = f"{'run':12s} {'overall':>8s}" + "".join(f"{b:>13s}" for b in BUCKET_NAMES)
    print(hdr)
    for label in [f"{a}_{p}" for a in ("cond", "blind") for p in pairs]:
        r = d["runs"][label]
        print(f"{label:12s} {r['top1_overall']*100:8.2f}" +
              "".join(f"{r['top1_by_bucket'][b]*100:13.2f}" for b in BUCKET_NAMES))

    print("\n--- Paired delta, conditioned - blind (pp) ---")
    print(f"{'pair':12s} {'overall':>8s}" + "".join(f"{b:>13s}" for b in BUCKET_NAMES))
    for p in pairs:
        e = d["paired_deltas"][p]
        print(f"{p:12s} {e['overall']*100:+8.2f}" +
              "".join(f"{e['by_bucket'][b]*100:+13.2f}" for b in BUCKET_NAMES))
    s = d["delta_summary"]
    print(f"{'MEAN':12s} {s['overall']['mean']*100:+8.2f}" +
          "".join(f"{s['by_bucket'][b]['mean']*100:+13.2f}" for b in BUCKET_NAMES))
    print(f"{'RANGE':12s} {s['overall']['range']*100:8.2f}" +
          "".join(f"{s['by_bucket'][b]['range']*100:13.2f}" for b in BUCKET_NAMES))

    print("\n--- Arm overlap ---")
    for scope, o in d["arm_overlap"].items():
        verdict = "OVERLAP" if o["overlaps"] else "NO OVERLAP"
        print(f"  {scope:12s} cond [{o['cond']['min']*100:.2f}, {o['cond']['max']*100:.2f}]  "
              f"blind [{o['blind']['min']*100:.2f}, {o['blind']['max']*100:.2f}]  "
              f"-> {verdict} (gap {o['gap']*100:+.2f} pp)")

    print("\n--- Monotonic bucket ordering (Comfortable < Moderate < Pressured < Critical) ---")
    for p in pairs:
        print(f"  {p}: {'HOLDS' if d['monotonicity'][p]['monotonic'] else 'FAILS'}")
    print(f"  -> holds in all 3 pairs: {d['monotonicity']['holds_in_all_pairs']}")

    print("\n--- Level spread vs delta spread (internal) ---")
    for scope, st in d["stability"].items():
        print(f"  {scope:12s} level range (cond {st['level_range_cond']*100:.2f} / "
              f"blind {st['level_range_blind']*100:.2f}) vs delta range "
              f"{st['delta_range']*100:.2f} pp -> {st['level_over_delta']:.1f}x")

    print("\n--- Thinking-time MAE (s) and WDL accuracy (%) ---")
    print(f"{'run':12s} {'MAE all':>9s} {'MAE crit':>9s} {'WDL':>8s}")
    for label in [f"{a}_{p}" for a in ("cond", "blind") for p in pairs]:
        r = d["runs"][label]
        print(f"{label:12s} {r['time_mae_overall_s']:9.2f} {r['time_mae_critical_s']:9.2f} "
              f"{r['wdl_accuracy']*100:8.2f}")


if __name__ == "__main__":
    main()
