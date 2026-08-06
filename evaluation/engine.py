"""Board reconstruction and Stockfish centipawn-loss scoring.

Shared by every move-quality evaluation: :mod:`evaluation.move_quality`,
:mod:`evaluation.counterfactual` and :mod:`evaluation.counterfactual_tempo`.

Two responsibilities:

*Board reconstruction* — parquet rows store the position canonicalized to the
side to move (see ``inference/encoding.py``), so recovering a real ``chess.Board``
means undoing the mirror on odd plies and restoring castling rights, en passant
square and move counters. :func:`canonical_move_to_actual_move` applies the same
correction to a move's squares.

*Engine scoring* — :func:`compute_cpls` evaluates a batch of (position, moves)
tasks with a pool of UCI engines, one per worker thread. CPL is the drop from the
engine's evaluation of the position to its evaluation after the candidate move,
both from the mover's point of view, clipped to ``cpl_cap``. Mate scores are
mapped to ``mate_cp`` so a forced mate does not dominate the average. Results are
cached per move within a position, so scoring several models on the same board
costs one extra ``analyse`` call per *distinct* move.

The paper scores at Stockfish 17, depth 12.
"""

from __future__ import annotations

import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from inference.encoding import PIECE_MAP

EPS = 1e-12

PROMO_CODE_TO_PIECE = {
    0: None,
    1: chess.QUEEN,
    2: chess.ROOK,
    3: chess.BISHOP,
    4: chess.KNIGHT,
}

# Piece-code -> (type, colour); codes 1..6 are the mover's, +6 the opponent's.
REVERSE_PIECE: dict[int, tuple[int, bool]] = {}
for _piece_type, _code in PIECE_MAP.items():
    REVERSE_PIECE[_code] = (_piece_type, chess.WHITE)
    REVERSE_PIECE[_code + 6] = (_piece_type, chess.BLACK)

_THREAD_STATE = threading.local()
_ENGINE_REGISTRY: list[chess.engine.SimpleEngine] = []
_ENGINE_REGISTRY_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# Board reconstruction
# ----------------------------------------------------------------------
def decode_canonical_board(board_encoding: list[int]) -> chess.Board:
    """Rebuild the canonical (mover-to-move-as-white) board from piece codes."""
    board = chess.Board.empty()
    for square, value in enumerate(board_encoding):
        piece_value = int(value)
        if piece_value == 0:
            continue
        piece_type, color = REVERSE_PIECE[piece_value]
        board.set_piece_at(square, chess.Piece(piece_type, color))
    board.turn = chess.WHITE
    return board


def reconstruct_actual_board(row: Any) -> chess.Board:
    """Recover the real board from a parquet row.

    ``row`` is any object exposing ``board_history``, ``move_ply``,
    ``castling_rights``, ``ep_square`` and ``halfmove_clock`` — callers pass a
    ``SimpleNamespace`` built from the parquet columns.
    """
    board = decode_canonical_board([int(value) for value in row.board_history[0]])
    if row.move_ply % 2 == 1:
        board = board.mirror()
    castling = []
    if row.castling_rights[0]:
        castling.append("K")
    if row.castling_rights[1]:
        castling.append("Q")
    if row.castling_rights[2]:
        castling.append("k")
    if row.castling_rights[3]:
        castling.append("q")
    board.set_castling_fen("".join(castling) if castling else "-")
    board.ep_square = row.ep_square if 0 <= int(row.ep_square) < 64 else None
    board.halfmove_clock = int(row.halfmove_clock)
    board.fullmove_number = (int(row.move_ply) // 2) + 1
    return board


def canonical_move_to_actual_move(from_sq: int, to_sq: int, promo_code: int,
                                  move_ply: int) -> chess.Move:
    """Undo the canonical mirror on a move's squares (odd plies only)."""
    if move_ply % 2 == 1:
        from_sq ^= 56
        to_sq ^= 56
    return chess.Move(int(from_sq), int(to_sq),
                      promotion=PROMO_CODE_TO_PIECE.get(int(promo_code)))


# ----------------------------------------------------------------------
# Policy scoring
# ----------------------------------------------------------------------
def softmax_last_dim(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return (exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + EPS)).astype(np.float32)


def score_legal_moves(
    move_logits: np.ndarray,
    promo_logits: np.ndarray,
    legal_moves: list[list[int]],
) -> list[tuple[tuple[int, int], float, int, int, int]]:
    """Score a row's legal moves, best first.

    Promotions add the log-probability of the promotion piece from the promotion
    head, so under-promotions are ranked against queening rather than tied.
    """
    promo_probs = softmax_last_dim(np.asarray(promo_logits, dtype=np.float32))
    scored: list[tuple[tuple[int, int], float, int, int, int]] = []
    for from_sq, to_sq, promo_code in legal_moves:
        base_idx = int(from_sq) * 64 + int(to_sq)
        score = float(move_logits[base_idx])
        if promo_code and 56 <= int(to_sq) <= 63:
            file_idx = int(to_sq) - 56
            promo_idx = int(promo_code) - 1
            if 0 <= promo_idx < 4:
                score += float(np.log(max(EPS, promo_probs[file_idx, promo_idx])))
        scored.append(((base_idx, int(promo_code)), score,
                       int(from_sq), int(to_sq), int(promo_code)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


# ----------------------------------------------------------------------
# Stockfish
# ----------------------------------------------------------------------
def score_to_cp(score: chess.engine.PovScore, *, mate_cp: int) -> int:
    relative = score.relative if hasattr(score, "relative") else score
    if relative.is_mate():
        mate_in = relative.mate()
        if mate_in is None:
            return int(mate_cp)
        return int(mate_cp if mate_in > 0 else -mate_cp)
    cp = relative.score()
    return 0 if cp is None else int(cp)


def resolve_engine_path(engine_path: str) -> str:
    """Resolve a Stockfish binary, falling back to the usual install locations."""
    candidates = [engine_path]
    if engine_path != "/usr/games/stockfish":
        candidates.append("/usr/games/stockfish")
    if engine_path != "stockfish":
        candidates.append("stockfish")
    for candidate in candidates:
        if candidate == "stockfish":
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
            continue
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve Stockfish binary from candidates: {candidates}",
    )


def configure_engine(engine: chess.engine.SimpleEngine, *, hash_mb: int,
                     threads: int) -> None:
    options = {"Hash": max(1, int(hash_mb)), "Threads": max(1, int(threads))}
    try:
        engine.configure(options)
    except Exception:
        pass


def close_registered_engines() -> None:
    global _ENGINE_REGISTRY
    with _ENGINE_REGISTRY_LOCK:
        engines = _ENGINE_REGISTRY
        _ENGINE_REGISTRY = []
    for engine in engines:
        try:
            engine.quit()
        except Exception:
            pass


def init_engine_worker(engine_path: str, engine_depth: int, mate_cp: int,
                       cpl_cap: int, engine_hash_mb: int,
                       engine_threads_per_worker: int) -> None:
    """Spawn one engine per worker thread and stash it in thread-local state."""
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    configure_engine(engine, hash_mb=engine_hash_mb,
                     threads=engine_threads_per_worker)
    _THREAD_STATE.engine = engine
    _THREAD_STATE.limit = chess.engine.Limit(depth=engine_depth)
    _THREAD_STATE.mate_cp = int(mate_cp)
    _THREAD_STATE.cpl_cap = int(cpl_cap)
    with _ENGINE_REGISTRY_LOCK:
        _ENGINE_REGISTRY.append(engine)


def evaluate_cpl_task(task: tuple[str, dict[str, str | None]]) -> dict[str, int | None]:
    """CPL for each named move on one position. ``None`` where a move is absent."""
    engine = getattr(_THREAD_STATE, "engine", None)
    limit = getattr(_THREAD_STATE, "limit", None)
    mate_cp = getattr(_THREAD_STATE, "mate_cp", None)
    cpl_cap = getattr(_THREAD_STATE, "cpl_cap", None)
    if engine is None or limit is None or mate_cp is None or cpl_cap is None:
        raise RuntimeError("Stockfish worker was not initialized")
    fen, move_map = task
    board = chess.Board(fen)

    base_info = engine.analyse(board, limit)
    base_score = base_info.get("score")
    if not isinstance(base_score, chess.engine.PovScore):
        return {name: None for name in move_map}
    base_cp = score_to_cp(base_score, mate_cp=mate_cp)

    cached_results: dict[str, int | None] = {}
    move_cache: dict[str, int | None] = {}
    for name, move_uci in move_map.items():
        if not move_uci:
            cached_results[name] = None
            continue
        if move_uci in move_cache:
            cached_results[name] = move_cache[move_uci]
            continue
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            move_cache[move_uci] = None
            cached_results[name] = None
            continue
        board_after_move = board.copy(stack=False)
        board_after_move.push(move)
        after_info = engine.analyse(board_after_move, limit)
        after_score = after_info.get("score")
        if not isinstance(after_score, chess.engine.PovScore):
            move_cache[move_uci] = None
            cached_results[name] = None
            continue
        after_cp_active_pov = -score_to_cp(after_score, mate_cp=mate_cp)
        move_cache[move_uci] = int(np.clip(base_cp - after_cp_active_pov,
                                           -cpl_cap, cpl_cap))
        cached_results[name] = move_cache[move_uci]
    return cached_results


def engine_identity(engine_path: str) -> str:
    """Name reported by the engine over UCI, e.g. ``"Stockfish 17.1"``.

    Absolute CPL values depend on the engine build, so every run prints this and
    callers record it next to their results. Paired comparisons are unaffected —
    all arms of a run are scored by the same engine at the same depth.
    """
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as exc:                      # pragma: no cover - env specific
        return f"<unavailable: {exc}>"
    try:
        return str(engine.id.get("name", "<unknown>"))
    finally:
        engine.quit()


def compute_cpls(
    tasks: list[tuple[str, dict[str, str | None]]],
    *,
    engine_path: str,
    engine_depth: int,
    mate_cp: int,
    cpl_cap: int,
    engine_workers: int,
    engine_hash_mb: int,
    engine_threads_per_worker: int,
) -> list[dict[str, int | None]]:
    """Score every task, one engine per worker. Order matches ``tasks``."""
    engine_path = resolve_engine_path(engine_path)
    print(f"[engine] {engine_identity(engine_path)}  ({engine_path}, "
          f"depth {engine_depth}, {engine_workers} worker(s))")

    if engine_workers <= 1:
        init_engine_worker(engine_path, engine_depth, mate_cp, cpl_cap,
                           engine_hash_mb, engine_threads_per_worker)
        try:
            progress = tqdm(tasks, desc="Evaluating Stockfish CPL", unit="pos")
            return [evaluate_cpl_task(task) for task in progress]
        finally:
            close_registered_engines()

    results: list[dict[str, int | None]] = []
    with ThreadPoolExecutor(
        max_workers=engine_workers,
        initializer=init_engine_worker,
        initargs=(engine_path, engine_depth, mate_cp, cpl_cap,
                  engine_hash_mb, engine_threads_per_worker),
    ) as executor:
        try:
            progress = tqdm(
                executor.map(evaluate_cpl_task, tasks, chunksize=64),
                total=len(tasks),
                desc="Evaluating Stockfish CPL",
                unit="pos",
            )
            for result in progress:
                results.append(result)
        finally:
            close_registered_engines()
    return results
