"""Smoke test: parquet-row encoding produces what the model consumes.

Covers the seam that every evaluation goes through — parquet row ->
``inference.encoding.make_model_batch_from_parquet_row`` -> ``Maia3Net`` — and
checks that the clock actually reaches the policy.

Run with ``python -m tests.smoke_encoding``.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from inference.encoding import (HISTORY_LEN, build_history_from_position,
                                canonicalize, encode_legal_moves,
                                make_model_batch_from_parquet_row)
from model import Maia3Net, CONFIG_MAIA3_23M

BATCH_KEYS = ("board_history", "time_history", "rep_flags", "castling",
              "ep_mask", "scalars", "legal_mask")

# 1. e4 e5 2. Nf3 Nc6 -- White to move at ply 4, so canonical == actual board.
MOVES = ["e2e4", "e7e5", "g1f3", "b8c6"]


def build_row(active_clock_s: float, time_history_s: list[float]) -> dict:
    """A parquet-shaped row for a real position a few moves into a game."""
    board, board_history, rep_flags = build_history_from_position(
        chess.Board(), MOVES)
    return dict(
        board_history=board_history,
        legal_moves=encode_legal_moves(canonicalize(board)),
        repetition_flags=rep_flags,
        time_history_s=time_history_s,
        castling_rights=[1, 1, 1, 1],
        ep_square=-1,
        active_elo=1800,
        opponent_elo=1750,
        move_ply=4,
        halfmove_clock=2,
        active_clock_s=active_clock_s,
        opponent_clock_s=180.0,
        active_inc_s=2.0,
        opponent_inc_s=2.0,
        base_time_s=180.0,
    )


def as_tensors(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    out = {}
    for k in BATCH_KEYS:
        v = np.asarray(batch[k])
        out[k] = (torch.from_numpy(v.astype(bool)) if k == "legal_mask"
                  else torch.from_numpy(v.astype(np.int64 if k == "board_history"
                                                 else np.float32)))
    return out


def main() -> None:
    print("Batch layout")
    comfortable = make_model_batch_from_parquet_row(
        **build_row(170.0, [3.0] * HISTORY_LEN))

    assert set(BATCH_KEYS) <= set(comfortable), sorted(comfortable)
    assert np.asarray(comfortable["board_history"]).shape == (1, HISTORY_LEN, 64)
    assert np.asarray(comfortable["time_history"]).shape == (1, HISTORY_LEN)
    assert np.asarray(comfortable["rep_flags"]).shape == (1, HISTORY_LEN)
    assert np.asarray(comfortable["castling"]).shape == (1, 4)
    assert np.asarray(comfortable["ep_mask"]).shape == (1, 64)
    assert np.asarray(comfortable["scalars"]).shape == (1, 9)
    assert np.asarray(comfortable["legal_mask"]).shape == (1, 4096)

    # Scalars 3/4/5/6/8 are RAW SECONDS and must survive encoding unscaled --
    # the layout contract shared with dataset.py._sample_to_tensors.
    scalars = np.asarray(comfortable["scalars"])[0]
    assert scalars[3] == 170.0, f"active clock mangled: {scalars[3]}"
    assert scalars[4] == 180.0, f"opponent clock mangled: {scalars[4]}"
    assert scalars[8] == 180.0, f"base time mangled: {scalars[8]}"

    board, _, _ = build_history_from_position(chess.Board(), MOVES)
    n_legal = int(np.asarray(comfortable["legal_mask"])[0].sum())
    assert n_legal == board.legal_moves.count(), (
        f"legal mask has {n_legal} moves, board has {board.legal_moves.count()}")
    print(f"  shapes OK, raw-second scalars preserved, {n_legal} legal moves")

    # Wiring check only: the model here is randomly initialised, so the
    # magnitude of the response is meaningless -- all that matters is that the
    # clock changes the policy at all, i.e. the signal is not dropped on the
    # way in. The size of the trained effect is what the paper measures.
    print("\nThe clock reaches the policy")
    critical = make_model_batch_from_parquet_row(
        **build_row(4.0, [0.5] * HISTORY_LEN))
    model = Maia3Net(dict(CONFIG_MAIA3_23M)).eval()
    with torch.no_grad():
        a = model(as_tensors(comfortable))[0]
        b = model(as_tensors(critical))[0]
    # Illegal moves are masked to -inf, so finiteness only holds on legal ones.
    legal = torch.from_numpy(np.asarray(comfortable["legal_mask"])[0].astype(bool))
    a, b = a[0][legal], b[0][legal]
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), "non-finite legal logits"
    assert not torch.allclose(a, b), (
        "policy identical at 170 s and 4 s remaining — temporal conditioning "
        "is not reaching the model")
    delta = float((a - b).abs().max())
    print(f"  policy responds to the clock (max |Δlogit| = {delta:.4f})")

    print("\nsmoke_encoding: PASSED")


if __name__ == "__main__":
    main()
