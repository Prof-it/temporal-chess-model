from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple

import numpy as np
import chess


HISTORY_LEN = 8
NUM_SQUARES = 64
NUM_POLICY_OUTPUTS = 4096
TRAIN_ELO_MIN = 1200
TRAIN_ELO_MAX = 2500

# Must match `process_pgn.py`
PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

# Known increment values the model was trained on (in seconds).
# Rare increments are clamped to the nearest known value to improve generalization.
KNOWN_INCREMENTS = [0, 2, 3, 5, 10]


def clamp_to_known_increment(inc_s: float) -> float:
    """Clamp an increment to the nearest known value for better model generalization."""
    return float(min(KNOWN_INCREMENTS, key=lambda x: abs(x - inc_s)))


def canonicalize(board: chess.Board) -> chess.Board:
    # Same as `process_pgn_v2.canonicalize(board, board.turn)`.
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> List[int]:
    tokens = [0] * NUM_SQUARES
    for square in range(NUM_SQUARES):
        piece = board.piece_at(square)
        if piece is None:
            continue
        val = PIECE_MAP[piece.piece_type]
        tokens[square] = val if piece.color == chess.WHITE else val + 6
    return tokens


def encode_legal_moves(board: chess.Board) -> List[Tuple[int, int, int]]:
    # promo: 0 none, 1 q, 2 r, 3 b, 4 n (matches process_pgn_v2.PROMOTION_MAP)
    promo_map = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
    out: List[Tuple[int, int, int]] = []
    for mv in board.legal_moves:
        out.append((mv.from_square, mv.to_square, promo_map.get(mv.promotion, 0)))
    return out


def build_history_from_position(
    start_board: chess.Board,
    moves_uci: Iterable[str],
    *,
    history_len: int = HISTORY_LEN,
) -> Tuple[chess.Board, List[List[int]], List[int]]:
    """Replays moves and returns (final_board, board_history, repetition_flags).

    `board_history` is a list of `history_len` encoded canonical boards, newest first,
    matching `process_pgn_v2`.
    """

    board = start_board.copy(stack=False)

    hist: Deque[List[int]] = deque(maxlen=history_len)
    raw: Deque[chess.Board] = deque(maxlen=history_len)

    # Initial position before any moves: include it (like training does per-ply).
    canonical = canonicalize(board)
    hist.appendleft(encode_board(canonical))
    raw.appendleft(canonical.copy(stack=False))

    for uci in moves_uci:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal move in position list: {uci}")
        board.push(mv)
        canonical = canonicalize(board)
        hist.appendleft(encode_board(canonical))
        raw.appendleft(canonical.copy(stack=False))

    board_history = list(hist)
    while len(board_history) < history_len:
        board_history.append([0] * NUM_SQUARES)
    board_history = board_history[:history_len]

    repetition = _repetition_flags(list(raw), history_len=history_len)
    return board, board_history, repetition


def _repetition_flags(history: List[chess.Board], *, history_len: int) -> List[int]:
    # Same idea as `process_pgn_v2.repetition_flags`.
    seen: dict[str, int] = {}
    flags: List[int] = []
    for b in history:
        key = b.board_fen()
        seen[key] = seen.get(key, 0) + 1
        flags.append(1 if seen[key] >= 2 else 0)
    while len(flags) < history_len:
        flags.insert(0, 0)
    return flags[:history_len]


@dataclass
class ContextOptions:
    active_elo: int = 1900
    opponent_elo: int = 1900
    active_clock_s: float = 300.0
    opponent_clock_s: float = 300.0
    active_inc_s: float = 0.0
    opponent_inc_s: float = 0.0
    # Optional: game base time (seconds) for time-control categorization.
    # Training uses the *initial* base time from PGN TimeControl, not the remaining clock.
    tc_base_s: float | None = None
    halfmove_clock: int = 0


def make_model_batch(
    *,
    board: chess.Board,
    board_history: List[List[int]],
    repetition_flags: List[int],
    time_history_s: Optional[List[float]] = None,
    ctx: ContextOptions,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Builds the exact numpy batch expected by the ONNX inference backend.

    Returns a dict of numpy arrays, each with a leading batch dimension of 1.

    Notes:
    - Mirrors training: board_history + legal_moves are canonicalized, but castling/ep/scalars
      are taken from the *actual* board (see `process_pgn_v2`).
    - `legal_mask` is 4096-wide (from-to moves only). Resign and flag are
      predicted by separate binary heads, not policy indices.
    """

    if time_history_s is None:
        time_history_s = [0.0] * HISTORY_LEN
    if len(time_history_s) != HISTORY_LEN:
        raise ValueError(f"time_history_s must have len {HISTORY_LEN}")

    # Canonical board for legal-move encoding.
    canonical = canonicalize(board)
    legal_moves = encode_legal_moves(canonical)

    legal_mask = np.zeros(NUM_POLICY_OUTPUTS, dtype=np.bool_)
    for from_sq, to_sq, _promo in legal_moves:
        legal_mask[from_sq * 64 + to_sq] = True

    # legal_mask is 4096-wide — policy outputs are from-to moves only.
    # Resign / flag are read from their own binary heads (model outputs
    # resign_logit / flag_logit), not from policy indices.

    # --- Scalars (must match dataset.py normalization) ---
    active_elo_norm = (ctx.active_elo - 1900) / 700.0
    opp_elo_norm = (ctx.opponent_elo - 1900) / 700.0
    ply_norm = board.fullmove_number * 2 - (0 if board.turn == chess.WHITE else 1)
    ply_norm = ply_norm / 100.0

    # Clamp remaining clock to tc_base_s when available to prevent
    # increment-inflated clocks from being out-of-distribution.
    _active_for_scalar = ctx.active_clock_s
    _opp_for_scalar = ctx.opponent_clock_s
    if ctx.tc_base_s is not None and ctx.tc_base_s > 0:
        _active_for_scalar = min(_active_for_scalar, ctx.tc_base_s)
        _opp_for_scalar = min(_opp_for_scalar, ctx.tc_base_s)
    # Clocks and increments now flow as RAW SECONDS through scalars[3..6] —
    # the time encoder featurizes them. ELO / ply / hmc stay normalized.
    active_clock_s = max(0.0, _active_for_scalar)
    opp_clock_s = max(0.0, _opp_for_scalar)

    # Increments flow as RAW SECONDS. The current time encoder (interp /
    # TimeControlToken) consumes increment continuously via log1p, and training
    # (dataset.py) feeds raw increments — so we must NOT quantize here. The old
    # clamp_to_known_increment() belonged to the legacy binned encoder and would
    # corrupt common controls (e.g. +1 -> 0); it is intentionally not applied.
    active_inc_s = float(ctx.active_inc_s)
    opp_inc_s = float(ctx.opponent_inc_s)
    hmc_norm = float(ctx.halfmove_clock) / 100.0

    # base_time (initial game seconds). If the caller didn't supply tc_base_s,
    # reconstruct an approximation from active_clock + 40*inc — matches the
    # original tc_cat formula and keeps the scalar in-distribution.
    if ctx.tc_base_s is not None and ctx.tc_base_s > 0:
        base_time_s = float(ctx.tc_base_s)
    else:
        base_time_s = float(ctx.active_clock_s) + 40.0 * active_inc_s
    base_time_s = max(0.0, base_time_s)

    # Layout MUST stay in sync with dataset.py _sample_to_tensors:
    #   0 elo  1 opp_elo  2 ply
    #   3 active_clock_s  4 opp_clock_s  5 active_inc_s  6 opp_inc_s  (RAW SECONDS)
    #   7 hmc  8 base_time_s  (RAW SECONDS)
    scalars_list = [
        active_elo_norm,
        opp_elo_norm,
        ply_norm,
        active_clock_s,
        opp_clock_s,
        active_inc_s,
        opp_inc_s,
        hmc_norm,
        base_time_s,
    ]

    castling_list = [
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ]

    ep_mask_arr = np.zeros(64, dtype=np.float32)
    if board.ep_square is not None:
        ep_mask_arr[int(board.ep_square)] = 1.0

    batch: dict[str, np.ndarray] = {
        "board_history": np.array(board_history, dtype=np.int64),
        # Raw seconds; the model's time encoder log-compresses internally.
        "time_history": np.array(time_history_s, dtype=np.float32),
        "rep_flags": np.array(repetition_flags, dtype=np.float32),
        "castling": np.array(castling_list, dtype=np.float32),
        "ep_mask": ep_mask_arr,
        "scalars": np.array(scalars_list, dtype=np.float32),
        "legal_mask": legal_mask,
    }
    # Add batch dimension.
    batch = {k: np.expand_dims(v, 0) for k, v in batch.items()}

    return batch


def _clamp_elo_to_training_range(elo: int) -> int:
    return max(TRAIN_ELO_MIN, min(TRAIN_ELO_MAX, int(elo)))


def make_model_batch_from_parquet_row(
    *,
    board_history: List[List[int]] | np.ndarray,
    legal_moves: List[Tuple[int, int, int]] | List[List[int]],
    repetition_flags: List[int] | np.ndarray,
    time_history_s: List[float] | np.ndarray,
    castling_rights: List[int] | np.ndarray,
    ep_square: int,
    active_elo: int,
    opponent_elo: int,
    move_ply: int,
    halfmove_clock: int,
    active_clock_s: float,
    opponent_clock_s: float,
    active_inc_s: float,
    opponent_inc_s: float,
    base_time_s: float | None = None,
    clamp_elos: bool = False,
) -> dict[str, np.ndarray]:
    """Build an ONNX batch directly from parquet row fields.

    This mirrors the training dataset path rather than reconstructing features
    from a canonical board.
    """

    board_history_arr = np.asarray(board_history, dtype=np.int64)
    if board_history_arr.shape != (HISTORY_LEN, NUM_SQUARES):
        raise ValueError(
            f"board_history must have shape {(HISTORY_LEN, NUM_SQUARES)}, got {board_history_arr.shape}"
        )

    time_hist_arr = np.asarray(time_history_s, dtype=np.float32)
    if time_hist_arr.shape != (HISTORY_LEN,):
        time_hist_arr = np.zeros(HISTORY_LEN, dtype=np.float32)
    # Raw seconds, most-recent-first; the model's time encoder log-compresses internally.
    time_hist_arr = time_hist_arr[::-1].copy()

    rep_flags_arr = np.asarray(repetition_flags, dtype=np.float32)
    if rep_flags_arr.shape != (HISTORY_LEN,):
        rep_flags_arr = np.zeros(HISTORY_LEN, dtype=np.float32)

    castling_arr = np.asarray(castling_rights, dtype=np.float32)
    if castling_arr.shape != (4,):
        raise ValueError(f"castling_rights must have shape (4,), got {castling_arr.shape}")

    ep_mask_arr = np.zeros(64, dtype=np.float32)
    if 0 <= int(ep_square) < 64:
        ep_mask_arr[int(ep_square)] = 1.0

    legal_mask = np.zeros(NUM_POLICY_OUTPUTS, dtype=np.bool_)
    for from_sq, to_sq, _promo in legal_moves:
        if 0 <= int(from_sq) < 64 and 0 <= int(to_sq) < 64:
            legal_mask[int(from_sq) * 64 + int(to_sq)] = True

    active_elo_val = _clamp_elo_to_training_range(active_elo) if clamp_elos else int(active_elo)
    opponent_elo_val = _clamp_elo_to_training_range(opponent_elo) if clamp_elos else int(opponent_elo)
    # base_time fallback for older parquet rows that lack the column:
    # reconstruct using the same tc_cat formula (base + 40*inc) used during preprocessing.
    if base_time_s is None:
        base_time_eff = float(active_clock_s) + 40.0 * float(active_inc_s)
    else:
        base_time_eff = float(base_time_s)

    # Time fields are RAW SECONDS (encoder featurizes downstream); ELO/ply/hmc normalized.
    scalars_arr = np.array([
        (active_elo_val - 1900) / 700.0,
        (opponent_elo_val - 1900) / 700.0,
        float(move_ply) / 100.0,
        max(0.0, float(active_clock_s)),
        max(0.0, float(opponent_clock_s)),
        float(active_inc_s),
        float(opponent_inc_s),
        float(halfmove_clock) / 100.0,
        max(0.0, base_time_eff),
    ], dtype=np.float32)

    batch: dict[str, np.ndarray] = {
        "board_history": board_history_arr,
        "time_history": time_hist_arr,
        "rep_flags": rep_flags_arr,
        "castling": castling_arr,
        "ep_mask": ep_mask_arr,
        "scalars": scalars_arr,
        "legal_mask": legal_mask,
    }
    return {k: np.expand_dims(v, 0) for k, v in batch.items()}
