"""Paired accuracy delta by rating band x time-pressure bucket.

Reproduces the lower block of the paper's Table I: the time-conditioned minus
time-blind top-1 delta split by player rating band, showing that the gain grows
monotonically with skill and, within each band, with time pressure.

Consumes the per-row parquets written by :mod:`evaluation.policy_probe` (they
carry ``game_id``, ``ply``, ``active_elo``, ``bucket_index`` and ``correct``),
so this is a CPU-only re-analysis of an existing evaluation pass — no GPU and no
model loading.

Two CI flavours are reported for every cell:

``position``  paired bootstrap over positions (:func:`evaluation.metrics.paired_delta_ci`)
``clustered`` bootstrap over games (:func:`evaluation.plot_maia3_comparison.clustered_paired_delta`),
              which respects the correlation between positions within a game

On the paper's May-2026 test set the two coincide exactly: the set samples one
position per game (1,500,029 positions from 1,500,029 distinct games), so every
game cluster is a singleton and clustering is a no-op. Both are still reported,
because the distinction does matter on any eval set with several plies per game.

Usage:
    python -m evaluation.rating_band_deltas \
        --cond eval_results/paper_2026/rows_cond23_true_testset2026.parquet \
        --blind eval_results/paper_2026/rows_agn23_true_testset2026.parquet \
        --out eval_results/paper_2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BUCKET_NAMES, RATING_BANDS, rating_band_index
from .metrics import paired_delta_ci
from .plot_maia3_comparison import clustered_paired_delta

_LOAD_COLUMNS = ["game_id", "ply", "active_elo", "bucket_index", "correct"]

_BAND_LABELS = tuple(f"{lo}-{hi}" for lo, hi in RATING_BANDS)


def load_rows(path: str) -> pd.DataFrame:
    """Load a policy_probe row file and attach the merge key and rating band."""
    df = pd.read_parquet(path, columns=_LOAD_COLUMNS)
    df["correct"] = df["correct"].astype(np.float32)
    df["key"] = df["game_id"].astype(str) + "#" + df["ply"].astype(str)
    df["band"] = df["active_elo"].map(rating_band_index).astype(np.int64)
    return df


def align(cond: pd.DataFrame, blind: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the two arms on position identity.

    The delta is only paired if both arms scored the same positions, so the join
    is on ``game_id``/``ply`` rather than on row order.
    """
    merged = cond.merge(
        blind[["key", "correct"]], on="key", suffixes=("_cond", "_blind"))
    if len(merged) != len(cond) or len(merged) != len(blind):
        raise SystemExit(
            f"row files do not cover the same positions: cond={len(cond)} "
            f"blind={len(blind)} shared={len(merged)}")
    return merged


def _cell_delta(merged: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Both CI flavours for one subset of rows."""
    if merged.empty:
        return {"n": 0}
    a = merged["correct_cond"].to_numpy(np.float64)
    b = merged["correct_blind"].to_numpy(np.float64)
    clustered = clustered_paired_delta(
        merged[["key", "game_id", "correct_cond"]].rename(
            columns={"correct_cond": "correct"}),
        merged[["key", "correct_blind"]].rename(columns={"correct_blind": "correct"}),
        seed=seed, n_boot=n_boot)
    return {
        "n": int(len(merged)),
        "n_games": int(merged["game_id"].nunique()),
        "position": paired_delta_ci(a, b, n_boot, seed),
        "clustered": clustered,
    }


def compute(merged: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Delta for every (band, bucket) cell plus per-band and per-bucket margins."""
    out: dict[str, dict] = {"overall": _cell_delta(merged, n_boot, seed), "bands": {}}

    for bi, label in enumerate(_BAND_LABELS):
        band_rows = merged[merged["band"] == bi]
        entry = {"overall": _cell_delta(band_rows, n_boot, seed)}
        for b, name in enumerate(BUCKET_NAMES):
            entry[name] = _cell_delta(band_rows[band_rows["bucket_index"] == b],
                                      n_boot, seed)
        out["bands"][label] = entry

    out["buckets"] = {
        name: _cell_delta(merged[merged["bucket_index"] == b], n_boot, seed)
        for b, name in enumerate(BUCKET_NAMES)
    }
    # Rows outside 1200-2500 carry band == -1; the paper's bands cover the test
    # set by construction, but report the count so a mismatch is visible.
    out["unbanded_rows"] = int((merged["band"] < 0).sum())
    return out


def _pp(cell: dict, ci: str) -> str:
    return f"{cell[ci]['value'] * 100:+.2f}"


def _pp_ci(cell: dict, ci: str) -> str:
    d = cell[ci]
    return (f"{d['value'] * 100:+.2f} {{\\scriptsize[{d['lo'] * 100:.2f},"
            f"{d['hi'] * 100:.2f}]}}")


def table_band_deltas(res: dict, ci: str = "position") -> str:
    """LaTeX for the Table I lower block."""
    rows = []
    for label in _BAND_LABELS:
        entry = res["bands"][label]
        cells = " & ".join(_pp(entry[name], ci) for name in BUCKET_NAMES)
        rows.append(f"Elo {label.replace('-', '--')} & "
                    f"{_pp_ci(entry['overall'], ci)} & {cells} \\\\")
    body = "\n".join(rows)
    return f"""% paired delta by rating band, {ci} bootstrap CIs
\\multicolumn{{6}}{{l}}{{\\textit{{Paired $\\Delta$ (pp) by rating band}}}} \\\\
\\hline
{body}
\\hline"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cond", required=True,
                   help="policy_probe row parquet for the time-conditioned arm")
    p.add_argument("--blind", required=True,
                   help="policy_probe row parquet for the time-blind arm")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap resamples per cell (0 disables CIs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ci", choices=("position", "clustered"), default="position",
                   help="Which CI flavour the emitted .tex quotes")
    p.add_argument("--out", default=None, help="Output dir for JSON + .tex")
    args = p.parse_args()

    cond, blind = load_rows(args.cond), load_rows(args.blind)
    merged = align(cond, blind)
    print(f"[bands] {len(merged):,} paired positions, "
          f"{merged['game_id'].nunique():,} games")

    res = compute(merged, n_boot=args.bootstrap, seed=args.seed)
    _print_summary(res)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"cond": args.cond, "blind": args.blind,
                   "n_positions": int(len(merged)), "bootstrap": args.bootstrap,
                   "seed": args.seed, "deltas": res}
        (out / "rating_band_deltas.json").write_text(json.dumps(payload, indent=2))
        (out / "table_1_rating_bands.tex").write_text(
            table_band_deltas(res, args.ci) + "\n")
        print(f"\n[bands] wrote rating_band_deltas.json, table_1_rating_bands.tex to {out}")


def _print_summary(res: dict) -> None:
    for ci in ("position", "clustered"):
        print(f"\n### paired delta by rating band (pp, 95% CI, {ci} bootstrap)")
        header = "  band        overall            " + "".join(
            f"{n[:9]:>11s}" for n in BUCKET_NAMES)
        print(header)
        for label in _BAND_LABELS:
            entry = res["bands"][label]
            o = entry["overall"][ci]
            cells = "".join(f"{entry[n][ci]['value'] * 100:>+11.2f}"
                            for n in BUCKET_NAMES)
            print(f"  {label:11s} {o['value'] * 100:+.2f} "
                  f"[{o['lo'] * 100:+.2f},{o['hi'] * 100:+.2f}]{cells}")
        o = res["overall"][ci]
        print(f"  {'all':11s} {o['value'] * 100:+.2f} "
              f"[{o['lo'] * 100:+.2f},{o['hi'] * 100:+.2f}]" + "".join(
                  f"{res['buckets'][n][ci]['value'] * 100:>+11.2f}"
                  for n in BUCKET_NAMES))


if __name__ == "__main__":
    main()
