"""Analysis + figure for the Maia-3 system comparison and clock-permutation probe.

Consumes the per-row parquets written by ``evaluation.eval_maia3_official`` and
``evaluation.policy_probe`` (row-aligned via game_id/ply), and produces:

  - accuracy by time-pressure bucket per model, with *game-clustered* bootstrap
    CIs (positions within a game are correlated; per-position bootstrap
    understates uncertainty);
  - paired per-position deltas between chosen model pairs, also game-clustered;
  - the headline figure: accuracy by bucket for Maia-3 79M vs our
    time-conditioned 79M with true and permuted clocks;
  - a LaTeX table over all supplied models.

Usage (defaults match scripts/run_maia3_comparison_queue.sh outputs):
    python -m evaluation.plot_maia3_comparison \
        --rows "Maia-3 79M"=eval_results/paper_2026/rows_maia3_79m_testset2026.parquet \
               "Marvin 79M"=eval_results/paper_2026/rows_cond79_true_testset2026.parquet \
               "Marvin 79M (permuted clocks)"=eval_results/paper_2026/rows_cond79_permute_testset2026.parquet \
        --pairs "Marvin 79M-Maia-3 79M" "Marvin 79M-Marvin 79M (permuted clocks)" \
        --out-dir eval_results/paper_2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BUCKET_NAMES, BUCKET_LABELS

N_BOOT = 1000


_LOAD_COLUMNS = ["game_id", "ply", "bucket_index", "correct", "nll"]


def load_rows(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=_LOAD_COLUMNS)
    df["correct"] = df["correct"].astype(np.float32)
    df["key"] = df["game_id"].astype(str) + "#" + df["ply"].astype(str)
    return df


def _cluster_boot(s: np.ndarray, n: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    """Bootstrap over game clusters in small chunks to cap peak memory."""
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    chunk = 100
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, len(s), size=(size, len(s)))
        boots[start:start + size] = (
            s[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1.0))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def clustered_accuracy_ci(df: pd.DataFrame, seed: int = 0,
                          n_boot: int = N_BOOT) -> dict:
    """Mean accuracy with a bootstrap CI clustered by game."""
    g = df.groupby("game_id", sort=False)["correct"].agg(["sum", "count"])
    s = g["sum"].to_numpy(np.float32)
    n = g["count"].to_numpy(np.float32)
    lo, hi = _cluster_boot(s, n, seed, n_boot)
    return {"value": float(s.sum() / n.sum()), "lo": lo, "hi": hi,
            "n": int(n.sum()), "n_games": len(s)}


def clustered_paired_delta(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0,
                           n_boot: int = N_BOOT) -> dict:
    """Game-clustered CI on mean(a.correct - b.correct) over shared rows."""
    m = a[["key", "game_id", "correct"]].merge(
        b[["key", "correct"]], on="key", suffixes=("_a", "_b"))
    diff = (m["correct_a"] - m["correct_b"]).astype(np.float32)
    g = diff.groupby(m["game_id"], sort=False).agg(["sum", "count"])
    s = g["sum"].to_numpy(np.float32)
    n = g["count"].to_numpy(np.float32)
    lo, hi = _cluster_boot(s, n, seed, n_boot)
    return {"value": float(s.sum() / n.sum()), "lo": lo, "hi": hi, "n": int(n.sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rows", nargs="+", required=True, help="label=path specs")
    p.add_argument("--pairs", nargs="*", default=[],
                   help="'labelA-labelB' paired-delta specs (A minus B)")
    p.add_argument("--fig-models", nargs="*", default=None,
                   help="Labels to draw in the figure (default: all)")
    p.add_argument("--out-dir", default="eval_results/paper_2026")
    p.add_argument("--tag", default="maia3_comparison")
    args = p.parse_args()

    models: dict[str, pd.DataFrame] = {}
    for spec in args.rows:
        label, path = spec.split("=", 1)
        models[label] = load_rows(path)
        print(f"[cmp] {label}: {len(models[label]):,} rows")

    results: dict = {"models": {}, "pairs": {}}
    for label, df in models.items():
        entry = {"overall": clustered_accuracy_ci(df)}
        for b, name in enumerate(BUCKET_NAMES):
            sub = df[df["bucket_index"] == b]
            entry[name] = clustered_accuracy_ci(sub) if len(sub) else None
        nll = df["nll"].dropna()
        entry["nll_mean"] = float(nll.mean())
        results["models"][label] = entry

    for spec in args.pairs:
        la, lb = spec.split("-", 1)
        a, b = models[la], models[lb]
        entry = {"overall": clustered_paired_delta(a, b)}
        for bi, name in enumerate(BUCKET_NAMES):
            entry[name] = clustered_paired_delta(
                a[a["bucket_index"] == bi], b[b["bucket_index"] == bi])
        results["pairs"][f"{la} minus {lb}"] = entry

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(results, indent=2))

    # ---- console summary -------------------------------------------------
    for label, e in results["models"].items():
        row = "  ".join(f"{name}={e[name]['value']*100:.2f}" for name in BUCKET_NAMES if e[name])
        print(f"[cmp] {label:38s} overall={e['overall']['value']*100:.2f} | {row}")
    for name, e in results["pairs"].items():
        row = "  ".join(f"{b}={e[b]['value']*100:+.2f}[{e[b]['lo']*100:+.2f},{e[b]['hi']*100:+.2f}]"
                        for b in BUCKET_NAMES)
        print(f"[cmp] Δ {name}: overall={e['overall']['value']*100:+.2f} | {row}")

    # ---- LaTeX table ------------------------------------------------------
    lines = [
        r"\begin{tabular}{lccccc}", r"\hline",
        r"\textbf{Model} & \textbf{Overall} & \textbf{Comf.} & \textbf{Mod.} & "
        r"\textbf{Press.} & \textbf{Crit.} \\",
        "  & " + " & ".join(f"({lbl})" for lbl in ("", *BUCKET_LABELS))[3:] + r" \\",
        r"\hline",
    ]
    for label, e in results["models"].items():
        cells = [f"{e['overall']['value']*100:.1f}"] + [
            f"{e[name]['value']*100:.1f}" if e[name] else "--" for name in BUCKET_NAMES]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_dir / f"table_{args.tag}.tex").write_text("\n".join(lines) + "\n")

    # ---- headline figure ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_labels = args.fig_models or list(models)
    colors = {lbl: c for lbl, c in zip(
        fig_labels, ["#c0392b", "#27ae60", "#2980b9", "#8e44ad", "#f39c12", "#333333"])}
    x = np.arange(len(BUCKET_NAMES))
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    for lbl in fig_labels:
        e = results["models"][lbl]
        vals = [e[name]["value"] * 100 for name in BUCKET_NAMES]
        los = [e[name]["lo"] * 100 for name in BUCKET_NAMES]
        his = [e[name]["hi"] * 100 for name in BUCKET_NAMES]
        style = "--" if "permut" in lbl.lower() else "-"
        ax.errorbar(x, vals, yerr=[np.subtract(vals, los), np.subtract(his, vals)],
                    marker="o", capsize=3, linestyle=style,
                    color=colors.get(lbl), label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({l})" for n, l in zip(BUCKET_NAMES, BUCKET_LABELS)])
    ax.set_ylabel("Top-1 move-matching accuracy (%)")
    ax.set_xlabel("Time-pressure bucket (clock remaining / base)")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_{args.tag}.png")
    print(f"[cmp] wrote {out_dir}/{{{args.tag}.json, table_{args.tag}.tex, fig_{args.tag}.png}}")


if __name__ == "__main__":
    main()
