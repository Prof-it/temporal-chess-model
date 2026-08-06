"""Per-row policy evaluation of our checkpoints with clock-input transforms.

Produces row-level parquet files (aligned with ``eval_maia3_official.py``
outputs via ``game_id``/``ply``) for the paper's system comparison and the
clock-permutation causal probe:

  - ``true``      — clocks as recorded (the standard evaluation).
  - ``permute``   — active/opponent clocks + move-time history are swapped in
                    from a random *donor row of the same time control* (same
                    base time and increment), with a fixed seed. This preserves
                    the marginal distribution of clock inputs while destroying
                    their association with the position: if the model's accuracy
                    edge flows through the clock, it must vanish here. Preferred
                    over zeroing, which the model reads as extreme time trouble
                    rather than absence of information.
  - ``frac:<f>``  — both clocks set to ``f * base_time`` (counterfactual
                    injection, matching evaluation/counterfactual.py); tempo
                    history left untouched.

Buckets in the output always reflect the row's REAL clock condition, so
"accuracy in the critical bucket under permuted clocks" reads as: how well the
model predicts genuinely time-pressured humans when its clock input is
uninformative.

Move scoring follows evaluation/metrics.py exactly (legal-move scores from the
from-to logits plus the promotion-bias head; correctness at the from-to level;
resign/flag rows skipped). NLL is the softmax over scored legal moves with the
exact human triplet as target.

Usage:
    python -m evaluation.policy_probe \
        --model cond79=checkpoints/chessformer_79m_elo-endpoints_time-tokens_tempotok_s67_latest.pt \
        --eval-set data/testset_2026-05 --transform permute \
        --out eval_results/paper_2026/rows_cond79_permute_testset2026.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .backends import make_backend
from .data import resolve_base_time, resolve_eval_files, bucket_index, _encode_row
from .metrics import _softmax_rows, EPS

_COLUMNS = [
    "game_id", "ply", "board_history", "legal_moves", "time_history", "active_elo",
    "opp_elo", "move_ply", "halfmove_clock", "repetition_flags", "castling_rights",
    "ep_square", "base_time", "tc_base_s", "tc_inc_s", "tc_cat", "active_clock",
    "opp_clock", "active_inc", "opp_inc", "y_val", "policy_move",
    "policy_is_resign", "policy_is_flag",
]

_HIST = 8


def _iter_rows(files, limit=None):
    seen = 0
    for path in files:
        pf = pq.ParquetFile(str(path))
        avail = set(pf.schema_arrow.names)
        cols = [c for c in _COLUMNS if c in avail]
        for rb in pf.iter_batches(batch_size=20_000, columns=cols):
            for row in rb.to_pylist():
                yield row
                seen += 1
                if limit is not None and seen >= limit:
                    return


def build_permutation(files, limit, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 1: donor clock values, permuted within (base_time, increment) cells.

    Returns (donor_active, donor_opp, donor_time_history) row-aligned with the
    streaming order of pass 2.
    """
    active, opp, hist, cells = [], [], [], []
    for row in _iter_rows(files, limit):
        active.append(float(row["active_clock"]))
        opp.append(float(row["opp_clock"]))
        th = list(row["time_history"])[:_HIST]
        hist.append(th + [0.0] * (_HIST - len(th)))
        cells.append((resolve_base_time(row), float(row["active_inc"])))

    active = np.asarray(active, dtype=np.float32)
    opp = np.asarray(opp, dtype=np.float32)
    hist = np.asarray(hist, dtype=np.float32)

    rng = np.random.default_rng(seed)
    donor = np.arange(len(active))
    cells_arr = np.asarray(cells)
    # Group row indices by cell and permute within each.
    order = np.lexsort((cells_arr[:, 1], cells_arr[:, 0]))
    sorted_cells = cells_arr[order]
    boundaries = np.flatnonzero(
        (np.diff(sorted_cells[:, 0]) != 0) | (np.diff(sorted_cells[:, 1]) != 0)) + 1
    for group in np.split(order, boundaries):
        donor[group] = rng.permutation(group)

    n_cells = len(boundaries) + 1
    moved = float((donor != np.arange(len(donor))).mean())
    print(f"[probe] permutation over {n_cells} TC cells; {moved*100:.1f}% of rows "
          f"received another row's clocks")
    return active[donor], opp[donor], hist[donor]


def _legal_scores(move_logits: np.ndarray, promo_logits: np.ndarray,
                  legal_moves: list) -> tuple[np.ndarray, list[int]]:
    """Score every legal move exactly as metrics._best_legal_move does."""
    promo_probs = _softmax_rows(np.asarray(promo_logits, dtype=np.float32))
    scores = np.empty(len(legal_moves), dtype=np.float64)
    fromto = []
    for j, (from_sq, to_sq, promo_code) in enumerate(legal_moves):
        idx = int(from_sq) * 64 + int(to_sq)
        s = float(move_logits[idx])
        if promo_code and 56 <= int(to_sq) <= 63:
            pi = int(promo_code) - 1
            if 0 <= pi < 4:
                s += float(np.log(max(EPS, promo_probs[int(to_sq) - 56, pi])))
        scores[j] = s
        fromto.append(idx)
    return scores, fromto


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="label=path (.pt or .onnx)")
    p.add_argument("--eval-set", required=True)
    p.add_argument("--transform", default="true",
                   help="'true', 'permute', or 'frac:<f>' (e.g. frac:1.0)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()

    label, path = args.model.split("=", 1)
    files, set_label = resolve_eval_files(args.eval_set)
    backend = make_backend(path, device=args.device, config_name=label)

    transform = args.transform
    frac = None
    donors = None
    if transform.startswith("frac:"):
        frac = float(transform.split(":", 1)[1])
        transform = "frac"
    elif transform == "permute":
        donors = build_permutation(files, args.limit, args.seed)
    elif transform != "true":
        raise SystemExit(f"Unknown transform {args.transform!r}")

    out_rows: dict[str, list] = {k: [] for k in (
        "game_id", "ply", "active_elo", "bucket_index", "clock_ratio",
        "injected_ratio", "tc_base_s", "tc_inc_s", "tc_cat", "correct", "nll",
        "wdl_correct", "n_legal", "target_from", "target_to", "pred_from", "pred_to",
    )}

    batch_inputs, batch_meta = [], []
    n_skipped = 0

    def _flush():
        nonlocal batch_inputs, batch_meta
        inputs = {k: np.stack([b[k] for b in batch_inputs])
                  for k in batch_inputs[0].keys()}
        outputs = backend(inputs, return_promo=True)
        (move_logits, wdl_logits, _resign, _flag, _tb, _tf, _tt, promo_logits) = outputs
        wdl_pred = wdl_logits.argmax(axis=1)

        for i, m in enumerate(batch_meta):
            scores, fromto = _legal_scores(move_logits[i], promo_logits[i], m["legal"])
            best = int(np.argmax(scores))
            logz = float(np.logaddexp.reduce(scores - scores.max()) + scores.max())
            nll = logz - float(scores[m["target_pos"]]) if m["target_pos"] is not None else float("nan")

            out_rows["game_id"].append(m["game_id"])
            out_rows["ply"].append(m["ply"])
            out_rows["active_elo"].append(m["elo"])
            out_rows["bucket_index"].append(m["bucket"])
            out_rows["clock_ratio"].append(m["ratio"])
            out_rows["injected_ratio"].append(m["injected_ratio"])
            out_rows["tc_base_s"].append(m["tc_base_s"])
            out_rows["tc_inc_s"].append(m["tc_inc_s"])
            out_rows["tc_cat"].append(m["tc_cat"])
            out_rows["correct"].append(bool(fromto[best] == m["target_idx"]))
            out_rows["nll"].append(nll)
            out_rows["wdl_correct"].append(bool(int(wdl_pred[i]) == m["y_val"]))
            out_rows["n_legal"].append(len(m["legal"]))
            out_rows["target_from"].append(m["target_idx"] // 64)
            out_rows["target_to"].append(m["target_idx"] % 64)
            out_rows["pred_from"].append(fromto[best] // 64)
            out_rows["pred_to"].append(fromto[best] % 64)
        batch_inputs, batch_meta = [], []

    from tqdm import tqdm
    progress = tqdm(desc=f"{label}/{args.transform} {set_label}", unit="pos")
    for ridx, row in enumerate(_iter_rows(files, args.limit)):
        if row.get("policy_is_resign") or row.get("policy_is_flag"):
            n_skipped += 1
            continue

        base_time = resolve_base_time(row)
        real_active = float(row["active_clock"])
        ratio = min(max(real_active / max(base_time, 1.0), 0.0), 1.0)

        # Apply the clock transform to the model INPUT only; bucket/ratio in
        # the output stay tied to the real condition.
        if transform == "permute":
            row = dict(row)
            row["active_clock"] = float(donors[0][ridx])
            row["opp_clock"] = float(donors[1][ridx])
            row["time_history"] = donors[2][ridx].tolist()
        elif transform == "frac":
            row = dict(row)
            row["active_clock"] = frac * base_time
            row["opp_clock"] = frac * base_time
        injected_ratio = min(max(float(row["active_clock"]) / max(base_time, 1.0), 0.0), 1.0)

        legal = [list(m) for m in row["legal_moves"]]
        pm = list(row["policy_move"])
        tf_, tt_, tp_ = int(pm[0]), int(pm[1]), int(pm[2])
        target_pos = next((j for j, m in enumerate(legal)
                           if int(m[0]) == tf_ and int(m[1]) == tt_ and int(m[2]) == tp_), None)

        batch_inputs.append(_encode_row(row))
        batch_meta.append({
            "game_id": row["game_id"], "ply": int(row["ply"]),
            "elo": int(row["active_elo"]), "bucket": bucket_index(ratio),
            "ratio": ratio, "injected_ratio": injected_ratio,
            "tc_base_s": float(row.get("tc_base_s") or base_time),
            "tc_inc_s": float(row.get("tc_inc_s") or row.get("active_inc") or 0.0),
            "tc_cat": int(row.get("tc_cat", -1)),
            "legal": legal, "target_idx": tf_ * 64 + tt_, "target_pos": target_pos,
            "y_val": int(round(float(row["y_val"]) * 2)),
        })
        progress.update(1)
        if len(batch_meta) >= args.batch_size:
            _flush()
    if batch_meta:
        _flush()
    progress.close()

    table = pa.table(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)

    correct = np.asarray(out_rows["correct"])
    buckets = np.asarray(out_rows["bucket_index"])
    print(f"\n[probe] {label} transform={args.transform} on {set_label}: "
          f"n={len(correct):,} (skipped resign/flag={n_skipped:,})")
    print(f"[probe] top-1 accuracy overall: {correct.mean()*100:.2f}%")
    for b, name in enumerate(("Comfortable", "Moderate", "Pressured", "Critical")):
        msk = buckets == b
        if msk.any():
            print(f"[probe]   {name:11s}: {correct[msk].mean()*100:.2f}%  (n={int(msk.sum()):,})")
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
