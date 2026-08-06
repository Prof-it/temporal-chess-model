"""Emit the paper's LaTeX tables from a run.py metrics JSON.

Table I  — top-1 move-matching accuracy by time-pressure bucket (time-conditioned
           vs time-agnostic, plus the paired delta with 95% CI).
Table II — behavioural heads (WDL accuracy, thinking-time MAE, resignation F1).

Usage:
    python -m evaluation.tables --metrics eval_results/run/metrics_val.json \
        --cond small --agnostic small_notime --out eval_results/run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import BUCKET_NAMES

_BUCKET_SHORT = ("Comf.", "Mod.", "Press.", "Crit.")
_BUCKET_RANGE = (r"$>$50\%", r"20--50\%", r"5--20\%", r"$<$5\%")


def _acc(model: dict, bucket: str) -> float:
    return model["accuracy_by_bucket"][bucket]["value"] * 100.0


def table_accuracy(cond: dict, agn: dict, paired: dict | None) -> str:
    """Table I upper block: overall plus the four time-pressure buckets.

    The overall column is the headline number, so it leads the row exactly as it
    does in the paper; the bucket columns follow in increasing pressure order.
    """
    def acc_row(m: dict) -> str:
        cells = [m["accuracy_overall"]["value"] * 100.0]
        cells += [_acc(m, b) for b in BUCKET_NAMES]
        return " & ".join(f"{c:.2f}" for c in cells)

    cond_row, agn_row = acc_row(cond), acc_row(agn)
    if paired is not None:
        deltas = [paired["overall"]["value"] * 100.0]
        deltas += [paired[b]["value"] * 100.0 for b in BUCKET_NAMES]
        ci = paired["overall"]
        ci_note = (f"\n95\\% CI (overall) & {{\\scriptsize[{ci['lo'] * 100:.2f},"
                   f"{ci['hi'] * 100:.2f}]}}" + " & " * len(BUCKET_NAMES) + "\\\\")
    else:
        deltas = [cond["accuracy_overall"]["value"] * 100.0
                  - agn["accuracy_overall"]["value"] * 100.0]
        deltas += [_acc(cond, b) - _acc(agn, b) for b in BUCKET_NAMES]
        ci_note = ""
    delta_row = " & ".join(f"{d:+.2f}" for d in deltas)

    cols = " & ".join(f"\\textbf{{{s}}}" for s in ("Overall",) + _BUCKET_SHORT)
    ranges = " & " + " & ".join(f"({r})" for r in _BUCKET_RANGE)
    return f"""\\begin{{table}}[t]
\\caption{{Top-1 move-matching accuracy (percent) by time-pressure bucket, 23M models.}}
\\label{{tab:accuracy}}
\\centering
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\textbf{{Model}} & {cols} \\\\
 {ranges} \\\\
\\hline
Time-conditioned & {cond_row} \\\\
Time-blind       & {agn_row} \\\\
\\hline
$\\Delta$ (pp)   & {delta_row} \\\\{ci_note}
\\hline
\\end{{tabular}}
\\end{{table}}"""


def table_aux(cond: dict, agn: dict) -> str:
    def wdl(m): return m["wdl"]["accuracy"] * 100.0
    def mae(m): return m["time_head"]["mae_seconds"] if m["time_head"] else float("nan")
    def f1(m): return m["resignation"]["f1"]
    return f"""\\begin{{table}}[t]
\\caption{{Behavioural heads, 23M models. The time-agnostic baseline keeps these heads but receives no clock signal.}}
\\label{{tab:aux}}
\\centering
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Head (metric)}} & \\textbf{{Time-cond.}} & \\textbf{{Time-agnostic}} \\\\
\\hline
Outcome, WDL accuracy (\\%) & {wdl(cond):.1f} & {wdl(agn):.1f} \\\\
Thinking time, MAE (s)     & {mae(cond):.2f} & {mae(agn):.2f} \\\\
Resignation, F1            & {f1(cond):.2f} & {f1(agn):.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""


def main() -> None:
    p = argparse.ArgumentParser(description="Emit paper LaTeX tables from metrics JSON")
    p.add_argument("--metrics", required=True, help="run.py metrics JSON")
    p.add_argument("--cond", required=True, help="label of the time-conditioned model")
    p.add_argument("--agnostic", required=True, help="label of the time-agnostic model")
    p.add_argument("--out", default=None, help="output dir for .tex files")
    args = p.parse_args()

    payload = json.loads(Path(args.metrics).read_text())
    models = payload["models"]
    cond, agn = models[args.cond], models[args.agnostic]
    paired = payload.get("paired_deltas", {}).get(f"{args.cond}_minus_{args.agnostic}")

    t1 = table_accuracy(cond, agn, paired)
    t2 = table_aux(cond, agn)
    print(t1 + "\n\n" + t2)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "table_1_accuracy.tex").write_text(t1 + "\n")
        (out / "table_2_aux.tex").write_text(t2 + "\n")
        print(f"\n[tables] wrote table_1_accuracy.tex, table_2_aux.tex to {out}")


if __name__ == "__main__":
    main()
