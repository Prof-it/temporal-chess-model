"""Gate: every paper checkpoint still loads and still scores what it published.

Run before and after any edit to model.py / train.py / dataset.py. Checkpoints
embed their own ``config_dict`` (written by train.py, read by
evaluation.backends), so deleting a config key the model no longer reads can
still break loading — or, worse, silently change a code path. This checks both.

    python -m tests.gate_checkpoints --checkpoints-dir checkpoints \
        --eval-set data/testset_2026-05 --limit 20000

Writes a JSON of per-checkpoint accuracies. Re-run with --baseline <that file>
to assert nothing moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from evaluation.backends import make_backend
from evaluation.data import iter_eval_batches, resolve_eval_files, BUCKET_NAMES
from evaluation.metrics import MetricsAccumulator

# label -> candidate paths relative to --checkpoints-dir; the first that exists
# wins. The released filename comes first, the internal training filename second,
# so the gate runs against either a downloaded release or a raw training dir.
PAPER_CHECKPOINTS = {
    "cond23_1M": ["marvin_23m_cond_1M.pt",
                  "archive/chessformer_maia3-23m_elo-endpoints_time-tokens_tempotok_s67_step1000000.pt"],
    "blind23_1M": ["marvin_23m_blind_1M.pt",
                   "archive/chessformer_23m_elo-endpoints_time-blind_s67_step1000000.pt"],
    "cond23_250k_s67": ["marvin_23m_cond_250k_s67.pt",
                        "archive/chessformer_maia3-23m_elo-endpoints_time-tokens_tempotok_s67_step250000.pt"],
    "blind23_250k_s67": ["marvin_23m_blind_250k_s67.pt",
                         "archive/chessformer_23m_elo-endpoints_time-blind_s67_step250000.pt"],
    "cond23_250k_s68": ["marvin_23m_cond_250k_s68.pt",
                        "archive/chessformer_23m_elo-endpoints_time-tokens_tempotok_s68_d43_step250000.pt"],
    "blind23_250k_s68": ["marvin_23m_blind_250k_s68.pt",
                         "archive/chessformer_23m_elo-endpoints_time-blind_s68_d43_step250000.pt"],
    "cond23_250k_s69": ["marvin_23m_cond_250k_s69.pt",
                        "archive/chessformer_23m_elo-endpoints_time-tokens_tempotok_s69_d44_step250000.pt"],
    "blind23_250k_s69": ["marvin_23m_blind_250k_s69.pt",
                         "archive/chessformer_23m_elo-endpoints_time-blind_s69_d44_step250000.pt"],
    "cond79": ["marvin_79m_cond.pt",
               "chessformer_79m_elo-endpoints_time-tokens_tempotok_s67_latest.pt"],
}

TOLERANCE = 1e-9   # bit-identical is the expectation; this is float slop only


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoints-dir", default="checkpoints")
    p.add_argument("--eval-set", default="data/testset_2026-05")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None, help="Write results JSON here")
    p.add_argument("--baseline", default=None,
                   help="Compare against a previous run's JSON and fail on drift")
    args = p.parse_args()

    root = Path(args.checkpoints_dir)
    present = {}
    for label, candidates in PAPER_CHECKPOINTS.items():
        for rel in candidates:
            if (root / rel).exists():
                present[label] = root / rel
                break
    missing = sorted(set(PAPER_CHECKPOINTS) - set(present))
    if missing:
        print(f"[gate] SKIPPING (not found): {', '.join(missing)}")
    if not present:
        raise SystemExit("[gate] no checkpoints found — nothing to verify")

    files, set_label = resolve_eval_files(args.eval_set)
    print(f"[gate] {len(present)} checkpoints x {args.limit:,} rows of {set_label}")

    results: dict[str, dict] = {}
    for label, path in present.items():
        # One model at a time: loading nine 23M/79M models at once exhausts GPU
        # memory, and a failure here should name the checkpoint that caused it.
        be = make_backend(str(path), device=args.device, config_name=label)
        acc = MetricsAccumulator(has_time_head=getattr(be, "has_time_head", True))
        for batch in iter_eval_batches(files, batch_size=args.batch_size,
                                       limit=args.limit, progress=False):
            acc.update(be(batch.inputs, return_promo=True), batch.metas)
        a = acc._arrays()
        entry = {
            "n": int(a["move_correct"].size),
            "top1": float(a["move_correct"].mean()),
            "by_bucket": {name: float(a["move_correct"][a["bucket"] == b].mean())
                          for b, name in enumerate(BUCKET_NAMES)},
            "wdl": float(a["wdl_correct"].mean()),
            "time_mae": (float(a["time_abs_err"].mean())
                         if a["time_abs_err"].size else None),
        }
        results[label] = entry
        print(f"  {label:18s} top1={entry['top1']*100:6.2f}  "
              f"wdl={entry['wdl']*100:6.2f}  "
              f"mae={entry['time_mae'] if entry['time_mae'] is None else round(entry['time_mae'], 3)}")
        del be

    payload = {"eval_set": set_label, "limit": args.limit, "results": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\n[gate] wrote {args.out}")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())["results"]
        drift = []
        for label, entry in results.items():
            if label not in base:
                drift.append(f"{label}: absent from baseline")
                continue
            for key in ("top1", "wdl"):
                d = abs(entry[key] - base[label][key])
                if d > TOLERANCE:
                    drift.append(f"{label}.{key}: {base[label][key]:.10f} -> "
                                 f"{entry[key]:.10f}  (Δ {d:.2e})")
            for name in BUCKET_NAMES:
                d = abs(entry["by_bucket"][name] - base[label]["by_bucket"][name])
                if d > TOLERANCE:
                    drift.append(f"{label}.{name}: Δ {d:.2e}")
        if drift:
            print("\n[gate] FAILED — outputs changed:")
            for line in drift:
                print("  " + line)
            raise SystemExit(1)
        print(f"\n[gate] PASSED — all {len(results)} checkpoints match the baseline")


if __name__ == "__main__":
    main()
