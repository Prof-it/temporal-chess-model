"""Run consolidated evaluation over one or more models on a shared eval set.

The eval set is encoded once and every model is run on each batch, so per-row
metrics arrays are row-aligned across models — enabling paired bootstrap deltas
(e.g. time vs no-time accuracy within each time-pressure bucket).

Example:
    python -m evaluation.run \
        --models cond=checkpoints/marvin_23m_cond_1M.pt \
                 agn=checkpoints/marvin_23m_blind_1M.pt \
        --eval-set data/testset_2026-05 --paired-with agn --out results

Model spec is ``label=path`` (``.pt`` or ``.onnx``). A ``.pt`` lacking a
``config_dict`` needs an explicit config and is not supported from the CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from .backends import make_backend
from .data import iter_eval_batches, resolve_eval_files, BUCKET_NAMES
from .metrics import MetricsAccumulator, paired_delta_ci


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidated model evaluation")
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more label=path specs (.pt or .onnx)")
    p.add_argument("--eval-set", required=True,
                   help="Directory of parquet shards (or a single .parquet)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of rows (for quick runs)")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap resamples for CIs (0 disables)")
    p.add_argument("--paired-with", default=None,
                   help="Label of the baseline model for paired accuracy deltas "
                        "(e.g. the no-time arm)")
    p.add_argument("--out", default=None, help="Output dir for metrics JSON")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _parse_model_specs(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--models entry must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out


def main() -> None:
    args = parse_args()
    model_specs = _parse_model_specs(args.models)
    files, set_label = resolve_eval_files(args.eval_set)
    print(f"[eval] set={set_label} shards={len(files)} models={[m[0] for m in model_specs]}")

    backends = {label: make_backend(path, device=args.device, config_name=label)
                for label, path in model_specs}
    accums = {label: MetricsAccumulator(has_time_head=getattr(be, "has_time_head", True))
              for label, be in backends.items()}

    for batch in iter_eval_batches(files, batch_size=args.batch_size, limit=args.limit):
        for label, be in backends.items():
            outputs = be(batch.inputs, return_promo=True)
            accums[label].update(outputs, batch.metas)

    results: dict[str, dict] = {}
    arrays: dict[str, dict] = {}
    for label, acc in accums.items():
        m = acc.finalize(bootstrap=args.bootstrap, seed=args.seed)
        arrays[label] = m.pop("_arrays")
        results[label] = m

    # Paired accuracy deltas vs a baseline model (row-aligned).
    paired = {}
    if args.paired_with and args.paired_with in arrays:
        base = arrays[args.paired_with]
        for label, arr in arrays.items():
            if label == args.paired_with:
                continue
            entry = {"overall": paired_delta_ci(
                arr["move_correct"], base["move_correct"], args.bootstrap, args.seed)}
            for b, name in enumerate(BUCKET_NAMES):
                mask = arr["bucket"] == b
                bmask = base["bucket"] == b
                entry[name] = paired_delta_ci(
                    arr["move_correct"][mask], base["move_correct"][bmask],
                    args.bootstrap, args.seed)
            paired[f"{label}_minus_{args.paired_with}"] = entry

    _print_summary(results, paired)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"eval_set": set_label, "n_shards": len(files),
                   "models": results, "paired_deltas": paired}
        out_path = out_dir / f"metrics_{set_label}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n[eval] wrote {out_path}")


def _fmt(ci: dict) -> str:
    return f"{ci['value']*100:.2f} [{ci['lo']*100:.2f},{ci['hi']*100:.2f}]"


def _print_summary(results: dict, paired: dict) -> None:
    print("\n" + "=" * 72)
    for label, m in results.items():
        print(f"\n### {label}  (n={m['n']:,})")
        print(f"  accuracy overall : {_fmt(m['accuracy_overall'])}")
        print("  by time bucket   : " + "  ".join(
            f"{name}={m['accuracy_by_bucket'][name]['value']*100:.2f}" for name in BUCKET_NAMES))
        print(f"  WDL accuracy     : {m['wdl']['accuracy']*100:.2f}   E[V] MSE={m['wdl']['ev_mse']:.4f}")
        if m["time_head"]:
            print(f"  time head        : bin_acc={m['time_head']['bin_accuracy']*100:.2f}%  "
                  f"MAE={m['time_head']['mae_seconds']:.2f}s")
        r = m["resignation"]
        print(f"  resignation      : P={r['precision']*100:.2f} R={r['recall']*100:.2f} "
              f"F1={r['f1']*100:.2f} acc={r['accuracy']*100:.2f} (support={r['support_positive']})")
    for name, entry in paired.items():
        print(f"\n### paired delta {name} (pp, 95% CI)")
        print(f"  overall: {entry['overall']['value']*100:+.2f} "
              f"[{entry['overall']['lo']*100:+.2f},{entry['overall']['hi']*100:+.2f}]")
        for bname in BUCKET_NAMES:
            e = entry[bname]
            print(f"  {bname:11s}: {e['value']*100:+.2f} [{e['lo']*100:+.2f},{e['hi']*100:+.2f}]")


if __name__ == "__main__":
    main()
