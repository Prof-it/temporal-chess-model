"""Metric accumulation for model evaluation.

A single streaming pass feeds model outputs + row metadata into
:class:`MetricsAccumulator`, which stores compact per-row arrays and finalizes
into the publication metrics:

- move-matching accuracy: overall, by Elo band, by time-pressure bucket, by band x bucket
- WDL: accuracy and E[V] MSE
- move-conditional time head: bin accuracy and MAE in seconds (current 0..CAP scheme)
- resignation: precision / recall / F1 / accuracy

Per-row storage lets us bootstrap confidence intervals on any aggregate or
between-model delta (see :func:`bootstrap_ci` and :func:`paired_delta_ci`).

Flagging is intentionally not reported.
"""

from __future__ import annotations

import numpy as np

from model import NUM_TIME_BINS, TIME_SECOND_CAP
from .data import RowMeta, BUCKET_NAMES, RATING_BANDS

EPS = 1e-12
SPECIAL_RESIGN_INDEX = 4096
SPECIAL_FLAG_INDEX = 4097


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=-1, keepdims=True) + EPS)


def _best_legal_move(move_logits: np.ndarray, promo_logits: np.ndarray,
                     legal_moves: list) -> tuple[int, float]:
    """Return (best_move_idx, best_score) over a row's legal moves.

    Promotion choice contributes an additive log-prob on rank-8 destinations,
    matching the model's promotion-bias head.
    """
    promo_probs = _softmax_rows(np.asarray(promo_logits, dtype=np.float32))  # (8,4)
    best_idx, best_score = -1, -np.inf
    for from_sq, to_sq, promo_code in legal_moves:
        idx = int(from_sq) * 64 + int(to_sq)
        score = float(move_logits[idx])
        if promo_code and 56 <= int(to_sq) <= 63:
            pi = int(promo_code) - 1
            if 0 <= pi < 4:
                score += float(np.log(max(EPS, promo_probs[int(to_sq) - 56, pi])))
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx, best_score


def _expected_ponder_seconds(time_logits: np.ndarray) -> float:
    """E[seconds] under the current scheme: bin index == seconds-spent (0..CAP)."""
    probs = _softmax_rows(time_logits[None])[0]
    bins = np.arange(NUM_TIME_BINS, dtype=np.float32)
    return float(np.sum(probs * bins))


class MetricsAccumulator:
    def __init__(self, has_time_head: bool = True):
        self.has_time_head = has_time_head
        self._move_correct: list[bool] = []
        self._bucket: list[int] = []
        self._band: list[int] = []
        self._elo: list[int] = []
        self._wdl_correct: list[bool] = []
        self._ev_pred: list[float] = []
        self._y_val: list[int] = []
        self._time_bin_correct: list[bool] = []
        self._time_top3_correct: list[bool] = []
        self._time_top5_correct: list[bool] = []
        self._time_abs_err: list[float] = []
        self._resign_pred: list[bool] = []
        self._resign_true: list[bool] = []

    def update(self, outputs: tuple, metas: list[RowMeta]) -> None:
        (move_logits, wdl_logits, resign_logit, _flag_logit,
         time_baseline, time_from_logits, time_to_logits, promo_logits) = outputs

        wdl_pred = wdl_logits.argmax(axis=1)
        wdl_prob = _softmax_rows(wdl_logits)
        ev = wdl_prob[:, 2] - wdl_prob[:, 0]   # P(win) - P(loss)

        for i, meta in enumerate(metas):
            # Resignation/flagging rows are excluded from move/time/wdl accuracy
            # (they are not regular board moves) but feed the resign classifier.
            best_idx, best_score = _best_legal_move(
                move_logits[i], promo_logits[i], meta.legal_moves)
            resign_score = float(resign_logit[i])
            flag_score = float(_flag_logit[i]) if _flag_logit is not None else -np.inf
            self._resign_pred.append(resign_score > best_score and resign_score >= flag_score)
            self._resign_true.append(meta.is_resign)

            if meta.is_resign or meta.is_flag:
                continue

            self._move_correct.append(best_idx == meta.target_move_idx)
            self._bucket.append(meta.bucket_index)
            self._band.append(meta.rating_band)
            self._elo.append(meta.active_elo)
            self._wdl_correct.append(int(wdl_pred[i]) == meta.y_val)
            self._ev_pred.append(float(ev[i]))
            self._y_val.append(meta.y_val)

            if self.has_time_head:
                tl = time_baseline[i] + time_from_logits[i, meta.policy_move[0]] \
                    + time_to_logits[i, meta.policy_move[1]]
                true_bin = min(int(round(meta.time_target_s)), TIME_SECOND_CAP)
                # Top-k bins by logit (argpartition on the last few for speed).
                topk = np.argpartition(tl, -5)[-5:]
                order = topk[np.argsort(tl[topk])[::-1]]
                self._time_bin_correct.append(int(order[0]) == true_bin)
                self._time_top3_correct.append(true_bin in order[:3])
                self._time_top5_correct.append(true_bin in order[:5])
                self._time_abs_err.append(abs(_expected_ponder_seconds(tl) - meta.time_target_s))

    # ------------------------------------------------------------------
    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "move_correct": np.asarray(self._move_correct, dtype=bool),
            "bucket": np.asarray(self._bucket, dtype=np.int64),
            "band": np.asarray(self._band, dtype=np.int64),
            "elo": np.asarray(self._elo, dtype=np.int64),
            "wdl_correct": np.asarray(self._wdl_correct, dtype=bool),
            "ev_pred": np.asarray(self._ev_pred, dtype=np.float64),
            "y_val": np.asarray(self._y_val, dtype=np.int64),
            "time_bin_correct": np.asarray(self._time_bin_correct, dtype=bool),
            "time_top3_correct": np.asarray(self._time_top3_correct, dtype=bool),
            "time_top5_correct": np.asarray(self._time_top5_correct, dtype=bool),
            "time_abs_err": np.asarray(self._time_abs_err, dtype=np.float64),
            "resign_pred": np.asarray(self._resign_pred, dtype=bool),
            "resign_true": np.asarray(self._resign_true, dtype=bool),
        }

    def finalize(self, bootstrap: int = 1000, seed: int = 0) -> dict:
        a = self._arrays()
        rng = np.random.default_rng(seed)
        n = a["move_correct"].size

        overall = mean_ci(a["move_correct"], bootstrap, rng)

        by_bucket = {}
        for b, name in enumerate(BUCKET_NAMES):
            mask = a["bucket"] == b
            by_bucket[name] = mean_ci(a["move_correct"][mask], bootstrap, rng)

        by_band = {}
        for bi, (lo, hi) in enumerate(RATING_BANDS):
            mask = a["band"] == bi
            by_band[f"{lo}-{hi}"] = mean_ci(a["move_correct"][mask], bootstrap, rng)

        # E[V] MSE: target in {-1,0,1} from y_val {0,1,2}
        ev_target = a["y_val"].astype(np.float64) - 1.0
        ev_mse = float(np.mean((a["ev_pred"] - ev_target) ** 2)) if n else float("nan")

        time_metrics = None
        if self.has_time_head and a["time_bin_correct"].size:
            time_metrics = {
                "bin_accuracy": float(a["time_bin_correct"].mean()),
                "top3_bin_accuracy": float(a["time_top3_correct"].mean()),
                "top5_bin_accuracy": float(a["time_top5_correct"].mean()),
                "mae_seconds": float(a["time_abs_err"].mean()),
                "medae_seconds": float(np.median(a["time_abs_err"])),
            }

        return {
            "n": int(n),
            "accuracy_overall": overall,
            "accuracy_by_bucket": by_bucket,
            "accuracy_by_rating_band": by_band,
            "wdl": {
                "accuracy": float(a["wdl_correct"].mean()) if n else float("nan"),
                "ev_mse": ev_mse,
            },
            "time_head": time_metrics,
            "resignation": binary_classification_metrics(a["resign_true"], a["resign_pred"]),
            "_arrays": a,   # retained for cross-model paired deltas; drop before JSON
        }


# ----------------------------------------------------------------------
def mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator,
            alpha: float = 0.05) -> dict:
    """Point estimate of the mean with a bootstrap percentile CI."""
    if x.size == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    value = float(x.mean())
    if n_boot <= 0:
        return {"value": value, "lo": value, "hi": value, "n": int(x.size)}
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": value, "lo": float(lo), "hi": float(hi), "n": int(x.size)}


def paired_delta_ci(x_a: np.ndarray, x_b: np.ndarray, n_boot: int, seed: int = 0,
                    alpha: float = 0.05) -> dict:
    """Bootstrap CI for mean(x_a) - mean(x_b) on row-aligned arrays.

    Use for A-vs-B deltas evaluated on the *same* positions (e.g. time vs
    no-time accuracy by bucket), which makes the comparison paired.
    """
    if x_a.size != x_b.size:
        raise ValueError(f"paired arrays must align: {x_a.size} vs {x_b.size}")
    if x_a.size == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    value = float(x_a.mean() - x_b.mean())
    if n_boot <= 0:
        return {"value": value, "lo": value, "hi": value, "n": int(x_a.size)}
    idx = rng.integers(0, x_a.size, size=(n_boot, x_a.size))
    boot = x_a[idx].mean(axis=1) - x_b[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": value, "lo": float(lo), "hi": float(hi), "n": int(x_a.size)}


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision / recall / F1 / accuracy for a binary classifier."""
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "support_positive": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
