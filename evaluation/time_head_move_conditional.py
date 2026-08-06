"""Does move-conditioning the think-time head carry real signal?

The paper's time head predicts a distribution over one-second bins for *each
candidate move* (position baseline + from-square + to-square terms). This
script quantifies what that buys, comparing three predictors of the human's
think time on the same positions:

  full      — the move-conditional distribution at the human's actual move
              (baseline + from + to). What the paper reports everywhere else.
  baseline  — the position-only baseline head, move terms ablated.
  mixture   — the policy-weighted average of the move-conditional distributions
              over all legal moves. This is the fair "position-only" ceiling:
              it uses the move terms but not the knowledge of which move was
              played. full < mixture (NLL) means knowing the move genuinely
              informs the timing prediction beyond the position.

Reports NLL of the true bin and expected-seconds MAE, overall and by
time-pressure bucket.

Usage:
    python -m evaluation.time_head_move_conditional \
        --model cond=checkpoints/archive/chessformer_maia3-23m_..._step1000000.pt \
        --eval-set data/testset_2026-05 --out eval_results/paper_2026/time_head_move_conditional.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backends import make_backend
from .data import iter_eval_batches, resolve_eval_files, BUCKET_NAMES
from .metrics import _softmax_rows, EPS


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True) + EPS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="label=path (.pt)")
    p.add_argument("--eval-set", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    label, path = args.model.split("=", 1)
    files, set_label = resolve_eval_files(args.eval_set)
    backend = make_backend(path, device=args.device, config_name=label)

    variants = ("full", "baseline", "mixture")
    nll = {v: [] for v in variants}
    err = {v: [] for v in variants}
    buckets: list[int] = []

    from model import NUM_TIME_BINS
    bins = np.arange(NUM_TIME_BINS, dtype=np.float64)

    for batch in iter_eval_batches(files, batch_size=args.batch_size, limit=args.limit):
        outputs = backend(batch.inputs, return_promo=True)
        (move_logits, _wdl, _res, _flag, time_baseline,
         time_from, time_to, promo_logits) = outputs

        for i, meta in enumerate(batch.metas):
            if meta.is_resign or meta.is_flag:
                continue
            true_bin = min(int(round(meta.time_target_s)), NUM_TIME_BINS - 1)
            fm, to = int(meta.policy_move[0]), int(meta.policy_move[1])

            # full: move-conditional at the human move.
            logits_full = time_baseline[i] + time_from[i, fm] + time_to[i, to]
            # baseline: position-only head, move terms ablated.
            logits_base = time_baseline[i]
            # mixture: policy-weighted average over legal moves' distributions.
            promo_probs = _softmax_rows(np.asarray(promo_logits[i], dtype=np.float32))
            scores, dists = [], []
            for from_sq, to_sq, promo_code in meta.legal_moves:
                idx = int(from_sq) * 64 + int(to_sq)
                s = float(move_logits[i][idx])
                if promo_code and 56 <= int(to_sq) <= 63:
                    pi = int(promo_code) - 1
                    if 0 <= pi < 4:
                        s += float(np.log(max(EPS, promo_probs[int(to_sq) - 56, pi])))
                scores.append(s)
                dists.append(time_baseline[i] + time_from[i, int(from_sq)]
                             + time_to[i, int(to_sq)])
            w = _softmax_rows(np.asarray(scores, dtype=np.float64)[None])[0]
            probs_mix = (w[:, None] * _softmax_rows(np.stack(dists))).sum(axis=0)

            for v, arr in (("full", None), ("baseline", None), ("mixture", probs_mix)):
                if v == "mixture":
                    probs = probs_mix
                else:
                    probs = _softmax_rows((logits_full if v == "full" else logits_base)[None])[0]
                nll[v].append(-float(np.log(max(probs[true_bin], EPS))))
                err[v].append(abs(float((probs * bins).sum()) - float(meta.time_target_s)))
            buckets.append(meta.bucket_index)

    buckets_arr = np.asarray(buckets)
    results: dict = {"model": label, "eval_set": set_label, "n": len(buckets)}
    for v in variants:
        nv, ev = np.asarray(nll[v]), np.asarray(err[v])
        entry = {"nll": float(nv.mean()), "mae_seconds": float(ev.mean()),
                 "by_bucket": {}}
        for b, name in enumerate(BUCKET_NAMES):
            m = buckets_arr == b
            entry["by_bucket"][name] = {"nll": float(nv[m].mean()),
                                        "mae_seconds": float(ev[m].mean()),
                                        "n": int(m.sum())}
        results[v] = entry
        print(f"[mc] {v:9s} NLL={entry['nll']:.4f}  MAE={entry['mae_seconds']:.3f}s  " +
              "  ".join(f"{name}={entry['by_bucket'][name]['nll']:.3f}"
                        for name in BUCKET_NAMES))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"[mc] wrote {out}")


if __name__ == "__main__":
    main()
