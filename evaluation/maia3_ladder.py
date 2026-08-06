"""Fairness ladder for the Maia-3 comparison on the 2026 test set.

Three reporting levels, each removing one of the baseline's protective
conditions (Chessformer trains blitz-only and discards positions after the
first sub-30s clock):

  L1 "Chessformer protocol" — blitz only, ply >= 10, positions strictly before
     either player first drops under 30 s (game-level cut). Their turf on
     clean, date-disjoint 2026 data; replaces the contaminated Allie anchor.
  L2 blitz, unfiltered      — same TC they trained on, time trouble included.
  L3 full set               — bullet + blitz + rapid (+ capped classical).

Uses the per-row parquets from eval_maia3_official / policy_probe; clocks are
joined from the source shards on (game_id, ply). Paired deltas are
game-clustered.

Usage:
    python -m evaluation.maia3_ladder \
        --baseline eval_results/paper_2026/rows_maia3_79m_testset2026.parquet \
        --ours eval_results/paper_2026/rows_cond79_true_testset2026.parquet \
        --eval-set data/testset_2026-05 --tag ladder_79m \
        --out-dir eval_results/paper_2026
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import BUCKET_NAMES
from .plot_maia3_comparison import clustered_accuracy_ci, clustered_paired_delta

# NOTE: test-shard `tc_cat` is a legacy game-duration category (<600s -> 0,
# <1800s -> 1, else 2), NOT the bullet/blitz taxonomy — blitz and bullet share
# category 0. Lichess (and Maia-3) categorize speed by estimated duration
# (base + 40*increment), which the shards' `tc_group` string encodes, so blitz
# is selected via tc_group joined from the source shards.


def load_clock_index(eval_set: str) -> pd.DataFrame:
    """(game_id, ply) -> tc_group + first sub-30s ply per game, from source shards."""
    frames = []
    for f in sorted(glob.glob(str(Path(eval_set) / "*.parquet"))):
        t = pq.read_table(f, columns=["game_id", "ply", "active_clock", "opp_clock",
                                      "tc_group"])
        frames.append(t.to_pandas())
    src = pd.concat(frames, ignore_index=True)
    below = (src["active_clock"] < 30.0) | (src["opp_clock"] < 30.0)
    sub30 = src.loc[below].groupby("game_id")["ply"].min().rename("first_sub30_ply")
    src = src.merge(sub30, on="game_id", how="left")
    src["first_sub30_ply"] = src["first_sub30_ply"].fillna(np.inf)
    return src[["game_id", "ply", "tc_group", "first_sub30_ply"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--baseline", required=True, help="Maia-3 per-row parquet")
    p.add_argument("--ours", required=True, help="our model's per-row parquet")
    p.add_argument("--eval-set", required=True)
    p.add_argument("--tag", default="ladder")
    p.add_argument("--out-dir", default="eval_results/paper_2026")
    args = p.parse_args()

    cols = ["game_id", "ply", "bucket_index", "correct"]
    base = pd.read_parquet(args.baseline, columns=cols)
    ours = pd.read_parquet(args.ours, columns=cols)
    for df in (base, ours):
        df["key"] = df["game_id"].astype(str) + "#" + df["ply"].astype(str)
        df["correct"] = df["correct"].astype(np.float32)

    clock_idx = load_clock_index(args.eval_set)
    base = base.merge(clock_idx, on=["game_id", "ply"], how="left")

    is_blitz = base["tc_group"] == "blitz"
    l1_keys = base.loc[
        is_blitz & (base["ply"] >= 10)
        & (base["ply"] < base["first_sub30_ply"]), "key"]
    l2_keys = base.loc[is_blitz, "key"]

    levels = {
        "L1 Chessformer protocol (blitz, ply>=10, pre-sub-30s)": set(l1_keys),
        "L2 blitz unfiltered": set(l2_keys),
        "L3 full set": None,
    }

    results = {}
    for name, keys in levels.items():
        b = base if keys is None else base[base["key"].isin(keys)]
        o = ours if keys is None else ours[ours["key"].isin(keys)]
        entry = {
            "n": int(len(b)),
            "baseline": clustered_accuracy_ci(b),
            "ours": clustered_accuracy_ci(o),
            "paired_delta": clustered_paired_delta(o, b),
        }
        if keys is None:  # per-bucket detail only for the full set
            entry["by_bucket"] = {
                bn: clustered_paired_delta(o[o["bucket_index"] == bi],
                                           b[b["bucket_index"] == bi])
                for bi, bn in enumerate(BUCKET_NAMES)}
        results[name] = entry
        d = entry["paired_delta"]
        print(f"[ladder] {name}: n={entry['n']:,}  "
              f"baseline={entry['baseline']['value']*100:.2f}  "
              f"ours={entry['ours']['value']*100:.2f}  "
              f"delta={d['value']*100:+.2f} [{d['lo']*100:+.2f},{d['hi']*100:+.2f}]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(results, indent=2))

    lines = [r"\begin{tabular}{lrccc}", r"\hline",
             r"\textbf{Evaluation regime} & \textbf{n} & \textbf{Maia-3} & "
             r"\textbf{Ours} & $\Delta$ (95\% CI) \\", r"\hline"]
    for name, e in results.items():
        d = e["paired_delta"]
        lines.append(
            f"{name.split('(')[0].strip()} & {e['n']:,} & "
            f"{e['baseline']['value']*100:.1f} & {e['ours']['value']*100:.1f} & "
            f"{d['value']*100:+.2f} [{d['lo']*100:+.2f},{d['hi']*100:+.2f}] \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_dir / f"table_{args.tag}.tex").write_text("\n".join(lines) + "\n")
    print(f"[ladder] wrote {out_dir}/{{{args.tag}.json, table_{args.tag}.tex}}")


if __name__ == "__main__":
    main()
