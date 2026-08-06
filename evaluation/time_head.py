"""In-depth evaluation of the move-conditional thinking-time head.

The time head predicts, for the human's actual move, a distribution over 121
one-second bins (bin index = seconds spent, capped at TIME_SECOND_CAP). This
module scores that prediction against the true ponder time and reports:

- MAE, MedAE, RMSE of E[seconds] vs actual seconds
- top-1 / top-3 / top-5 bin accuracy
- all of the above overall, by time-pressure bucket, and by Elo band
- mean predicted vs mean actual ponder time by bucket (does the head predict
  shorter thinks under pressure?)

Comparing the time-conditioned model against the time-blind baseline isolates
how much the clock signal helps thinking-time prediction. Outputs a metrics
JSON, a LaTeX table, and figures.

Usage:
    python -m evaluation.time_head --models cond=<tokens>.pt agn=<time-blind>.pt \
        --eval-set data/val --limit 200000 --out eval_results/paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model import NUM_TIME_BINS, TIME_SECOND_CAP
from .backends import make_backend
from .data import iter_eval_batches, resolve_eval_files, BUCKET_NAMES, RATING_BANDS

_BINS = np.arange(NUM_TIME_BINS, dtype=np.float64)
RATING_BAND_LABELS = [f"{lo}-{hi}" for lo, hi in RATING_BANDS]


def _softmax_rows(x):
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=-1, keepdims=True) + 1e-12)


class TimeHeadAccumulator:
    """Per-row thinking-time predictions vs ground truth, tagged by bucket/elo."""

    def __init__(self):
        self.pred_s: list[float] = []
        self.true_s: list[float] = []
        self.top1: list[bool] = []
        self.top3: list[bool] = []
        self.top5: list[bool] = []
        self.bucket: list[int] = []
        self.band: list[int] = []

    def update(self, outputs, metas):
        (_mv, _wdl, _rs, _fl, time_baseline, time_from, time_to, _promo) = outputs
        for i, m in enumerate(metas):
            if m.is_resign or m.is_flag:
                continue
            tl = time_baseline[i] + time_from[i, m.policy_move[0]] + time_to[i, m.policy_move[1]]
            probs = _softmax_rows(tl[None])[0]
            pred = float(np.sum(probs * _BINS))
            true_bin = min(int(round(m.time_target_s)), TIME_SECOND_CAP)
            topk = np.argpartition(tl, -5)[-5:]
            order = topk[np.argsort(tl[topk])[::-1]]
            self.pred_s.append(pred)
            self.true_s.append(float(m.time_target_s))
            self.top1.append(int(order[0]) == true_bin)
            self.top3.append(true_bin in order[:3])
            self.top5.append(true_bin in order[:5])
            self.bucket.append(m.bucket_index)
            self.band.append(m.rating_band)

    def arrays(self):
        return {k: np.asarray(getattr(self, k)) for k in
                ("pred_s", "true_s", "top1", "top3", "top5", "bucket", "band")}


def _metrics(pred, true, t1, t3, t5) -> dict:
    if pred.size == 0:
        return {"n": 0}
    err = np.abs(pred - true)
    return {
        "n": int(pred.size),
        "mae": float(err.mean()),
        "medae": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "top1": float(t1.mean()),
        "top3": float(t3.mean()),
        "top5": float(t5.mean()),
        "mean_pred_s": float(pred.mean()),
        "mean_true_s": float(true.mean()),
    }


def summarize(acc: TimeHeadAccumulator) -> dict:
    a = acc.arrays()
    out = {"overall": _metrics(a["pred_s"], a["true_s"], a["top1"], a["top3"], a["top5"])}
    by_bucket = {}
    for b, name in enumerate(BUCKET_NAMES):
        m = a["bucket"] == b
        by_bucket[name] = _metrics(a["pred_s"][m], a["true_s"][m], a["top1"][m], a["top3"][m], a["top5"][m])
    out["by_bucket"] = by_bucket
    by_band = {}
    for bi, lab in enumerate(RATING_BAND_LABELS):
        m = a["band"] == bi
        by_band[lab] = _metrics(a["pred_s"][m], a["true_s"][m], a["top1"][m], a["top3"][m], a["top5"][m])
    out["by_rating_band"] = by_band
    return out


# ---------------------------------------------------------------------------
def latex_table(cond: dict, agn: dict) -> str:
    o_c, o_a = cond["overall"], agn["overall"]
    return f"""\\begin{{table}}[t]
\\caption{{Move-conditional thinking-time head, 23M models. Metrics on the human's
actual move; the time-blind baseline keeps the head but sees no clock.}}
\\label{{tab:timehead}}
\\centering
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Time-cond.}} & \\textbf{{Time-blind}} \\\\
\\hline
MAE (s)            & {o_c['mae']:.2f} & {o_a['mae']:.2f} \\\\
MedAE (s)          & {o_c['medae']:.2f} & {o_a['medae']:.2f} \\\\
RMSE (s)           & {o_c['rmse']:.2f} & {o_a['rmse']:.2f} \\\\
Top-1 bin acc (\\%) & {o_c['top1']*100:.1f} & {o_a['top1']*100:.1f} \\\\
Top-3 bin acc (\\%) & {o_c['top3']*100:.1f} & {o_a['top3']*100:.1f} \\\\
Top-5 bin acc (\\%) & {o_c['top5']*100:.1f} & {o_a['top5']*100:.1f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""


def _figures(summaries: dict[str, dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"cond": "#2ca02c", "agn": "#d62728"}
    labels = list(summaries)
    x = np.arange(len(BUCKET_NAMES))

    # (1) MAE by bucket, per model.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = 0.8 / len(labels)
    for k, lab in enumerate(labels):
        mae = [summaries[lab]["by_bucket"][b]["mae"] for b in BUCKET_NAMES]
        ax.bar(x + (k - (len(labels) - 1) / 2) * w, mae, w, label=lab,
               color=colors.get(lab, None))
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_NAMES)
    ax.set_ylabel("Thinking-time MAE (s)")
    ax.set_title("Thinking-time prediction error by time pressure", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "timehead_mae_by_bucket.png", dpi=200); plt.close(fig)

    # (2) Mean predicted vs actual ponder time by bucket.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    actual = [summaries[labels[0]]["by_bucket"][b]["mean_true_s"] for b in BUCKET_NAMES]
    ax.plot(x, actual, "o-", color="#333333", linewidth=2.5, label="Actual (human)")
    for lab in labels:
        pred = [summaries[lab]["by_bucket"][b]["mean_pred_s"] for b in BUCKET_NAMES]
        ax.plot(x, pred, "o--", color=colors.get(lab, None), linewidth=2.2, label=f"{lab} predicted")
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_NAMES)
    ax.set_ylabel("Mean ponder time (s)")
    ax.set_title("Predicted vs actual thinking time by time pressure", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "timehead_pred_vs_actual.png", dpi=200); plt.close(fig)
    print(f"[time_head] wrote figures to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="In-depth thinking-time head evaluation")
    p.add_argument("--models", nargs="+", required=True, help="label=path specs")
    p.add_argument("--eval-set", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cond", default="cond", help="label of the time-conditioned model (for the table)")
    p.add_argument("--agnostic", default="agn", help="label of the time-blind model (for the table)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    specs = [s.split("=", 1) for s in args.models]
    files, set_label = resolve_eval_files(args.eval_set)
    backends = {lab.strip(): make_backend(path.strip(), device=args.device, config_name=lab.strip())
                for lab, path in specs}
    accs = {lab: TimeHeadAccumulator() for lab in backends}

    for batch in iter_eval_batches(files, batch_size=args.batch_size, limit=args.limit):
        for lab, be in backends.items():
            accs[lab].update(be(batch.inputs, return_promo=True), batch.metas)

    summaries = {lab: summarize(acc) for lab, acc in accs.items()}

    for lab, s in summaries.items():
        o = s["overall"]
        print(f"\n### {lab} (n={o['n']:,})")
        print(f"  MAE={o['mae']:.2f}s MedAE={o['medae']:.2f}s RMSE={o['rmse']:.2f}s "
              f"| top1={o['top1']*100:.1f}% top3={o['top3']*100:.1f}% top5={o['top5']*100:.1f}%")
        print("  MAE by bucket: " + "  ".join(
            f"{b}={s['by_bucket'][b]['mae']:.2f}" for b in BUCKET_NAMES))
        print("  pred/actual s by bucket: " + "  ".join(
            f"{b}={s['by_bucket'][b]['mean_pred_s']:.1f}/{s['by_bucket'][b]['mean_true_s']:.1f}" for b in BUCKET_NAMES))

    if args.cond in summaries and args.agnostic in summaries:
        print("\n" + latex_table(summaries[args.cond], summaries[args.agnostic]))

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / f"time_head_{set_label}.json").write_text(json.dumps(summaries, indent=2))
        if args.cond in summaries and args.agnostic in summaries:
            (out / "table_3_timehead.tex").write_text(latex_table(summaries[args.cond], summaries[args.agnostic]) + "\n")
        _figures(summaries, out)
        print(f"[time_head] wrote metrics + table to {out}")


if __name__ == "__main__":
    main()
