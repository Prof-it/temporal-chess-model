"""Camera-ready figures for the GEM 2026 paper, rendered at true IEEE column
width (3.5 in) with 8 pt fonts so text is readable without scaling.

Fig A  (quality.png): two stacked panels.
  (a) Sampled-move CPL gap to the human curve by time-pressure bucket, paired
      per position with bootstrap CIs. The human curve is the zero line.
  (b) Counterfactual: blunder rate vs injected clock fraction on fixed
      positions (the causal probe), human actual rate as reference.

Fig B  (comparison.png): accuracy by bucket — released Maia-3 79M, our
  time-conditioned 79M, and the same model with permuted clocks (game-clustered
  CIs, from maia3_comparison.json).

Usage:
    python -m evaluation.paper_figures --results-dir results --out-dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results")
BUCKETS = ["Comfortable", "Moderate", "Pressured", "Critical"]
BUCKET_SHORT = ["Comfortable\n(>50%)", "Moderate\n(20–50%)", "Pressured\n(5–20%)", "Critical\n(<5%)"]
GREEN, RED, BLUE, BLACK = "#1a9641", "#d7191c", "#2b83ba", "#222222"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.4, "lines.markersize": 4,
    "errorbar.capsize": 2.5, "legend.frameon": False,
})


def _boot_ci(x: np.ndarray, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def fig_quality(out: Path, cpl_parquet: Path, cf_parquet: Path,
                ylim: tuple[float, float]) -> None:
    df = pd.read_parquet(cpl_parquet)
    clean = lambda s: np.maximum(s.to_numpy(np.float64), 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.4), dpi=300)

    # (a) paired CPL gap to human, bars.
    x = np.arange(len(BUCKETS))
    width = 0.36
    for off, (label, col, color) in zip(
            (-width / 2, width / 2),
            (("Time-conditioned", "cond_cpl", GREEN), ("Time-blind", "agn_cpl", RED))):
        means, lo_e, hi_e = [], [], []
        for bi in range(4):
            s = df[df.bucket_index == bi].dropna(subset=["human_cpl", col])
            d = clean(s[col]) - clean(s["human_cpl"])
            m, lo, hi = _boot_ci(d, seed=bi)
            means.append(m); lo_e.append(m - lo); hi_e.append(hi - m)
        ax1.bar(x + off, means, width, yerr=[lo_e, hi_e], color=color,
                label=label, error_kw=dict(lw=0.9, capsize=2.5))
    ax1.axhline(0, color=BLACK, lw=1.0)
    ax1.text(0.02, 0.05, "0 = human move quality", transform=ax1.transAxes,
             fontsize=7, color=BLACK)
    ax1.set_xticks(x); ax1.set_xticklabels(BUCKET_SHORT)
    ax1.set_ylabel(r"$\Delta$CPL vs. human (cp)")
    ax1.set_title("(a) Gap to the human quality curve", fontsize=8.5)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12))
    ax1.grid(axis="y", alpha=0.3, lw=0.5)
    ax1.set_axisbelow(True)

    # (b) counterfactual on REAL pressured/critical positions: human baseline,
    # both models at the real clock, and cond with comfortable temporal context
    # (clock AND recent tempo) injected on the same boards.
    cfp = pd.read_parquet(cf_parquet)

    def rate(col, b, seed):
        v = cfp[cfp.bucket == b][col].dropna().to_numpy(np.float64)
        ind = (np.maximum(v, 0.0) > 300).astype(np.float64)
        m, lo, hi = _boot_ci(ind, seed=seed)
        return m * 100, (m - lo) * 100, (hi - m) * 100

    series = [("Human", "cpl_human", BLACK, None),
              ("Time-conditioned", "cpl_cond_real", GREEN, None),
              ("Time-blind", "cpl_agn_real", RED, None),
              ("Time-cond., comfortable\nclock+tempo injected", "cpl_cond_comfclk_tempo", GREEN, "//")]
    xb = np.arange(2)
    width = 0.2
    for i, (label, col, color, hatch) in enumerate(series):
        ms, lo_e, hi_e = zip(*[rate(col, b, seed=i) for b in (2, 3)])
        ax2.bar(xb + (i - 1.5) * width, ms, width, yerr=[lo_e, hi_e], label=label,
                color=color, hatch=hatch, alpha=0.55 if hatch else 1.0,
                error_kw=dict(lw=0.9, capsize=2.5))
    n_press = int((cfp.bucket == 2).sum())
    n_crit = int((cfp.bucket == 3).sum())
    ax2.set_xticks(xb)
    ax2.set_xticklabels([f"Pressured (5–20%)\nn={n_press:,}", f"Critical (<5%)\nn={n_crit:,}"])
    ax2.set_ylabel("Blunder rate (% > 300 cp)")
    ax2.set_ylim(*ylim)
    ax2.set_title("(b) Real time-trouble positions:\nblunders are clock-caused", fontsize=8.5)
    ax2.legend(loc="upper center", ncol=2, fontsize=7, columnspacing=0.8,
               handlelength=1.2, handletextpad=0.4)
    ax2.grid(axis="y", alpha=0.3, lw=0.5)
    ax2.set_axisbelow(True)

    fig.tight_layout(h_pad=1.6)
    fig.savefig(out / "quality.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_figures] wrote {out}/quality.png")


def fig_comparison(out: Path, comparison_json: Path,
                   ylim: tuple[float, float]) -> None:
    data = json.loads(Path(comparison_json).read_text())["models"]
    series = [
        ("Maia-3 79M (released)", "Maia-3 79M", RED, "-"),
        ("Ours (79M)", "Marvin 79M", GREEN, "-"),
        ("Ours, clock+tempo permuted", "Marvin 79M (permuted clocks)", BLUE, "--"),
    ]
    x = np.arange(len(BUCKETS))
    fig, ax = plt.subplots(figsize=(3.5, 2.7), dpi=300)
    for label, key, color, ls in series:
        e = data[key]
        vals = [e[b]["value"] * 100 for b in BUCKETS]
        lo = [(e[b]["value"] - e[b]["lo"]) * 100 for b in BUCKETS]
        hi = [(e[b]["hi"] - e[b]["value"]) * 100 for b in BUCKETS]
        ax.errorbar(x, vals, yerr=[lo, hi], marker="o", color=color,
                    linestyle=ls, label=label)
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_SHORT)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_figures] wrote {out}/comparison.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    p.add_argument("--results-dir", default=str(RES),
                   help="Directory holding the pipeline outputs; supplies the "
                        "defaults for --cpl-parquet/--cf-parquet/--comparison-json")
    p.add_argument("--cpl-parquet", default=None,
                   help="move_quality output (default: <results-dir>/cpl_sampled_tcbal_big.parquet)")
    p.add_argument("--cf-parquet", default=None,
                   help="counterfactual_tempo output (default: <results-dir>/counterfactual_tempo.parquet)")
    p.add_argument("--comparison-json", default=None,
                   help="plot_maia3_comparison output (default: <results-dir>/metrics/maia3_comparison.json)")
    p.add_argument("--out-dir", default="results/figures")
    # Axis limits are pinned so the published figures are byte-reproducible;
    # widen them if a rerun lands outside these ranges.
    p.add_argument("--blunder-ylim", nargs=2, type=float, default=(8.0, 13.6),
                   metavar=("LO", "HI"))
    p.add_argument("--accuracy-ylim", nargs=2, type=float, default=(53.6, 62.6),
                   metavar=("LO", "HI"))
    args = p.parse_args()

    res = Path(args.results_dir)
    cpl = Path(args.cpl_parquet or res / "cpl_sampled_tcbal_big.parquet")
    cf = Path(args.cf_parquet or res / "counterfactual_tempo.parquet")
    cmp_json = Path(args.comparison_json or res / "metrics" / "maia3_comparison.json")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_quality(out, cpl, cf, tuple(args.blunder_ylim))
    fig_comparison(out, cmp_json, tuple(args.accuracy_ylim))


if __name__ == "__main__":
    main()
