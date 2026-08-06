from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import chess

from inference.chessformer_policy import PolicyOutput
from inference.encoding import ContextOptions, build_history_from_position, canonicalize, make_model_batch
from inference.mcts import MCTSSettings, MCTSResult, mcts_choose_move, find_subtree_by_move_sequence, _Node
from inference.sampling import sample_from_logits

if TYPE_CHECKING:
    from inference.runtime import InferenceBackend


PROMO_INDEX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}


def _get_bot_elo(settings: dict) -> int:
    return int(settings.get("active_elo", settings.get("engine_elo", 1800)))


def _get_opponent_elo(settings: dict) -> int:
    return int(settings.get("opponent_elo", settings.get("human_elo", 1800)))


@dataclass(frozen=True)
class EffectivePolicyMove:
    move: chess.Move
    prob: float


def dynamic_temperature(settings: dict, current_ply: int) -> float:
    """Compute temperature with smooth exponential decay from temp_start → temp_min.

    Replaces the old step-function opening temperature logic.  The decay curve
    is: T(ply) = temp_min + (temp_start - temp_min) * exp(-decay_rate * ply).

    When temp_start / temp_min are **not** set (web app path), both default to
    ``settings["temperature"]`` and this function returns the slider value as-is.
    The decay only activates in the UCI engine path where configs set them.

    Settings used:
        temp_start      – initial (high) temperature at ply 0  (default: settings["temperature"])
        temp_min        – floor temperature to decay towards   (default: settings["temperature"])
        temp_decay_rate – controls how quickly decay happens   (default: 0.10)
                          ~0.10 → half-life ≈ 7 plies, ~0.05 → half-life ≈ 14 plies
    """
    base_temp = float(settings["temperature"])
    temp_start = float(settings.get("temp_start", base_temp))
    temp_min = float(settings.get("temp_min", base_temp))
    decay_rate = float(settings.get("temp_decay_rate", 0.10))

    if temp_start <= temp_min or decay_rate <= 0:
        return temp_min

    return temp_min + (temp_start - temp_min) * math.exp(-decay_rate * current_ply)


def _mirror_square(sq: int) -> int:
    return sq ^ 56


def _canonical_to_real_move(move: chess.Move, real_turn: chess.Color) -> chess.Move:
    if real_turn == chess.WHITE:
        return move
    return chess.Move(_mirror_square(move.from_square), _mirror_square(move.to_square), promotion=move.promotion)


def _sample_effective_policy_move(
    *,
    board: chess.Board,
    top_moves: list[dict],
    rng: np.random.Generator,
) -> EffectivePolicyMove | None:
    if not top_moves:
        return None

    probs = np.asarray([max(0.0, float(entry.get("prob", 0.0))) for entry in top_moves], dtype=np.float64)
    total = float(probs.sum())
    if total <= 0.0:
        probs = np.full(len(top_moves), 1.0 / len(top_moves), dtype=np.float64)
    else:
        probs = probs / total

    chosen_index = int(rng.choice(len(top_moves), p=probs))
    chosen = top_moves[chosen_index]
    move_uci = chosen.get("uci")
    if not isinstance(move_uci, str) or not move_uci:
        return None

    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None

    return EffectivePolicyMove(move=move, prob=float(probs[chosen_index]))


# --- numpy helpers ---

def _as_numpy(x: np.ndarray) -> np.ndarray:
    """Ensure numpy float32 array."""
    return np.asarray(x, dtype=np.float32)


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax over numpy array."""
    x = x.astype(np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    """1-D softmax returning numpy array."""
    a = _as_numpy(x).ravel()
    return _softmax_np(a)


NUM_TIME_BINS = 121  # Must match model.py NUM_TIME_BINS (TIME_SECOND_CAP + 1)
LEGACY_TIME_BINS = 128  # old sqrt-fraction-of-clock scheme


def _time_bin_to_seconds(
    bin_idx: int,
    active_clock_s: float,
    time_budget_cap: float | None = None,
    num_bins: int = NUM_TIME_BINS,
) -> float:
    """Decode a time-prediction bin index back to seconds.

    The scheme is selected by the model's time-head width ``num_bins`` (it is
    self-describing, so a legacy checkpoint keeps decoding correctly):

      - ``num_bins == 128`` (legacy): sqrt-fraction-of-clock bins —
        ``seconds = ((bin+0.5)/128)^2 * clock``. Intrinsically clock-scaled, so
        ``active_clock_s`` / ``time_budget_cap`` still apply here.
      - otherwise (current): linear absolute-second bins, 1 bin = 1 second; the
        top bin (``num_bins-1``) is the open-ended ">=CAP s" overflow. Returned
        as the raw predicted seconds with **no clock clamping** — the decoded
        time reflects the model's prediction, not the remaining clock. Flag
        safety is handled separately at the sleep site.

    *active_clock_s* / *time_budget_cap*: only used by the legacy clock-scaled
    scheme; ignored for the linear scheme.
    """
    if num_bins == LEGACY_TIME_BINS:
        clk = active_clock_s
        if time_budget_cap is not None and time_budget_cap > 0:
            clk = min(clk, time_budget_cap)
        scaled_mid = (bin_idx + 0.5) / LEGACY_TIME_BINS
        return float((scaled_mid**2) * max(1e-6, clk))
    # Linear raw-second bins: bin index == seconds. No clock clamping.
    return float(max(0.0, float(bin_idx)))


def _assemble_move_time_logits(
    t_baseline: np.ndarray,
    t_from: np.ndarray,
    t_to: np.ndarray,
    from_sq: int,
    to_sq: int,
) -> np.ndarray:
    """Assemble move-conditional time logits for a single move.

    time_logits(move) = baseline + time_from[from_sq] + time_to[to_sq]

    Args:
        t_baseline: (T,) position-level baseline logits
        t_from: (64, T) per-square from-move time bias
        t_to: (64, T) per-square to-move time bias
        from_sq: canonical from-square index (0-63)
        to_sq: canonical to-square index (0-63)

    Returns:
        (T,) move-conditional time bin logits
    """
    return t_baseline + t_from[from_sq] + t_to[to_sq]


def analyze_position(
    *,
    backend: "InferenceBackend",
    settings: dict,
    rng: np.random.Generator,
    board: chess.Board,
    moves_uci: list[str],
    active_clock_s: float,
    opponent_clock_s: float,
    active_inc_s: float = 0.0,
    opponent_inc_s: float = 0.0,
    time_history_s: list[float] | None = None,
    tc_base_s: float | None = None,
    initial_fen: str = chess.STARTING_FEN,
    skip_move_values: bool = False,
    skip_resign_flag: bool = False,
) -> dict:
    """Analyse the current position and return policy/value/time statistics.

    If *skip_move_values* is True the per-move forward passes are skipped and
    ``top_moves[i]['value']`` will be None for all moves.  The caller can
    fetch enriched stats later via a separate /eval_moves request.

    If *skip_resign_flag* is True the resign/flag head outputs are not read and
    the returned ``resign`` / ``flag`` probabilities and comparison fields will
    be zero / false. Resign and flag are dedicated binary heads in the same
    forward pass, so this no longer saves an NN evaluation — it just suppresses
    the analysis fields when the caller doesn't need them.
    """

    device = backend.device

    _final_board, board_history, repetition_flags = build_history_from_position(chess.Board(initial_fen), moves_uci)

    bot_elo = _get_bot_elo(settings)
    opponent_elo = _get_opponent_elo(settings)

    if board.turn == settings["human_color"]:
        active_elo = opponent_elo
        opp_elo = bot_elo
    else:
        active_elo = bot_elo
        opp_elo = opponent_elo

    ctx = ContextOptions(
        active_elo=active_elo,
        opponent_elo=opp_elo,
        active_clock_s=active_clock_s,
        opponent_clock_s=opponent_clock_s,
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        tc_base_s=(
            float(tc_base_s)
            if tc_base_s is not None
            else (float(settings.get("start_clock_s")) if settings.get("start_clock_s") is not None else None)
        ),
        halfmove_clock=int(board.halfmove_clock),
    )

    batch = make_model_batch(
        board=board,
        board_history=board_history,
        repetition_flags=repetition_flags,
        time_history_s=time_history_s,
        ctx=ctx,
        device=device,
    )

    # Cap for time decoding: don't let inflated clocks (increment accumulation)
    # produce OOD time predictions.  Use the game's base time as upper bound.
    _time_budget_cap: float | None = tc_base_s

    (m_logits, v_cls, resign_logit_t, flag_logit_t,
     t_baseline_raw, t_from_raw, t_to_raw, p_logits) = backend(batch, return_promo=True)

    m_logits = _as_numpy(m_logits[0])
    promo_p = _softmax_np(_as_numpy(p_logits[0]), axis=-1)

    # Time head components (position-level baseline + per-square biases)
    t_baseline = _as_numpy(t_baseline_raw[0])  # (T,)
    t_from = _as_numpy(t_from_raw[0])           # (64, T)
    t_to = _as_numpy(t_to_raw[0])               # (64, T)
    # Bin-decode scheme is self-describing via the head width (128 = legacy
    # sqrt-fraction; else linear raw-second). Thread to every _time_bin_to_seconds.
    num_time_bins = int(t_baseline.shape[-1])

    # Effective Policy
    canonical_board = canonicalize(board)
    legal_moves_data = []

    for mv in canonical_board.legal_moves:
        base_idx = mv.from_square * 64 + mv.to_square
        logit = float(m_logits[base_idx])
        if mv.promotion is not None and 56 <= mv.to_square <= 63:
            file_idx = mv.to_square - 56
            p_idx = PROMO_INDEX.get(mv.promotion, 0)
            promo_prob = float(promo_p[file_idx, p_idx])
            logit += float(np.log(max(1e-8, promo_prob)))
        legal_moves_data.append({"move": mv, "logit": logit})

    if not legal_moves_data:
        policy_display = []
    else:
        T = max(1e-4, float(settings["temperature"]))
        logits_vec = np.array([x["logit"] for x in legal_moves_data], dtype=np.float64)
        logits_vec = logits_vec / T
        probs_vec = _softmax_np(logits_vec.astype(np.float32))

        sorted_indices = np.argsort(-probs_vec)
        sorted_probs = probs_vec[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        target_p = float(settings["top_p"])
        cutoff_index = int(np.searchsorted(cumulative_probs, target_p))
        cutoff_count = min(len(sorted_probs), int(cutoff_index) + 1)
        cutoff_count = max(1, int(cutoff_count))

        kept_indices = sorted_indices[:cutoff_count]
        kept_probs = sorted_probs[:cutoff_count]
        kept_probs = kept_probs / kept_probs.sum()

        policy_display = []
        real_turn = board.turn
        for i, prob in zip(kept_indices.tolist(), kept_probs.tolist()):
            mv = legal_moves_data[i]["move"]
            real_mv = _canonical_to_real_move(mv, real_turn)
            # Assemble per-move time distribution
            mv_time_logits = _assemble_move_time_logits(
                t_baseline, t_from, t_to, mv.from_square, mv.to_square
            )
            try:
                label = board.san(real_mv)
            except Exception:
                label = real_mv.uci()
            policy_display.append({
                "label": label, "prob": prob, "uci": real_mv.uci(),
                "time_logits": mv_time_logits,
            })

    # ── Per-move value evaluation ──
    if policy_display and not skip_move_values:
        # Use the top move's time distribution to estimate expected move time
        # for constructing the child's time_history context.
        top_time_logits = policy_display[0].get("time_logits")
        if top_time_logits is not None:
            t_probs_pre = _softmax_np(top_time_logits)
            bins_pre = np.arange(len(t_probs_pre), dtype=np.float32)
            expected_bin_pre = float(np.dot(t_probs_pre, bins_pre))
            expected_time_pre = _time_bin_to_seconds(int(round(expected_bin_pre)), active_clock_s, _time_budget_cap, num_time_bins)
        else:
            expected_time_pre = 0.0
        # Prepend this time to the parent's history (newest first, drop oldest).
        child_time_history_s = [expected_time_pre] + list(time_history_s[:-1]) if time_history_s else [expected_time_pre] + [0.0] * 7

        child_moves_uci = list(moves_uci)  # will be extended per child
        for entry in policy_display:
            try:
                child_board = board.copy()
                child_board.push(chess.Move.from_uci(entry["uci"]))
                child_uci = child_moves_uci + [entry["uci"]]
                _cb, child_hist, child_rep = build_history_from_position(
                    chess.Board(initial_fen), child_uci
                )
                # In the child position, the side-to-move is the opponent.
                # Swap elo / clocks accordingly.
                child_ctx = ContextOptions(
                    active_elo=opp_elo,
                    opponent_elo=active_elo,
                    active_clock_s=opponent_clock_s,
                    opponent_clock_s=active_clock_s,
                    active_inc_s=float(opponent_inc_s) if opponent_inc_s else 0.0,
                    opponent_inc_s=float(active_inc_s) if active_inc_s else 0.0,
                    tc_base_s=ctx.tc_base_s,
                    halfmove_clock=int(child_board.halfmove_clock),
                )
                child_batch = make_model_batch(
                    board=child_board,
                    board_history=child_hist,
                    repetition_flags=child_rep,
                    time_history_s=child_time_history_s,
                    ctx=child_ctx,
                    device=device,
                )
                (_, cv_cls, *_cr) = backend(child_batch, return_promo=False)
                cv_wdl = _softmax_1d(cv_cls[0])  # [L, D, W]
                child_win = float(cv_wdl[2])  # Win probability
                # child_win is P(child side wins).  Our value = 1 − child_win.
                entry["value"] = round(1.0 - child_win, 4)
            except Exception:
                entry["value"] = None

    wdl = _softmax_1d(v_cls[0])

    # ── Move-conditional time distributions ──
    # Each candidate move now has its own time distribution.
    # We compute per-move stats and also select the top move's distribution
    # for the global time panel / time sampling.
    T_time = max(1e-4, float(settings.get("time_temperature", 1.0)))
    target_time_p = float(settings.get("time_top_p", 0.95))

    for entry in policy_display:
        mv_t_logits = entry.get("time_logits")
        if mv_t_logits is None:
            entry["time_dist"] = []
            entry["expected_time_s"] = 0.0
            entry["mode_time_s"] = 0.0
            continue

        mv_t_probs_raw = _softmax_np(mv_t_logits)
        mv_t_logits_scaled = mv_t_logits / T_time
        mv_t_probs = _softmax_np(mv_t_logits_scaled)

        # Mode (Argmax on raw probs)
        mode_bin = int(np.argmax(mv_t_probs_raw))
        entry["mode_time_s"] = _time_bin_to_seconds(mode_bin, active_clock_s, _time_budget_cap, num_time_bins)

        # Top-P filtered distribution for display
        t_sorted_idx = np.argsort(mv_t_probs)[::-1]
        t_cumsum = np.cumsum(mv_t_probs[t_sorted_idx])
        t_cutoff = int(np.searchsorted(t_cumsum, target_time_p) + 1)
        t_active_idx = np.sort(t_sorted_idx[:t_cutoff])

        mv_time_dist = []
        exp_t = 0.0
        mass = 0.0
        for idx in t_active_idx:
            sec = _time_bin_to_seconds(int(idx), active_clock_s, _time_budget_cap, num_time_bins)
            prob = float(mv_t_probs[idx])
            mv_time_dist.append({"sec": sec, "prob": prob})
            exp_t += prob * sec
            mass += prob
        entry["time_dist"] = mv_time_dist
        entry["expected_time_s"] = exp_t / mass if mass > 0 else 0.0

        # Drop the raw logits array from the returned dict (not JSON-serializable)
        del entry["time_logits"]

    # ── Global time sampling (uses top move's distribution) ──
    # Pick the top move's time distribution for the position-level time sample
    # that drives simulated thinking time and the time panel.
    top_entry = policy_display[0] if policy_display else None
    if top_entry and top_entry.get("time_dist"):
        # Reconstruct the top move's full logits for sampling
        top_mv = None
        for ld in legal_moves_data:
            real_mv = _canonical_to_real_move(ld["move"], board.turn)
            if real_mv.uci() == top_entry["uci"]:
                top_mv = ld["move"]
                break
        if top_mv is not None:
            top_t_logits = _assemble_move_time_logits(
                t_baseline, t_from, t_to, top_mv.from_square, top_mv.to_square
            )
        else:
            top_t_logits = t_baseline  # fallback

        top_t_probs_raw = _softmax_np(top_t_logits)
        mode_bin = int(np.argmax(top_t_probs_raw))
        mode_time_s = _time_bin_to_seconds(mode_bin, active_clock_s, _time_budget_cap, num_time_bins)
        expected_time_s = top_entry["expected_time_s"]
        time_dist = top_entry["time_dist"]

        if settings.get("use_mode_time", False):
            time_sample_s = mode_time_s
            time_sample_prob = float(top_t_probs_raw[mode_bin])
        elif settings.get("use_expected_time", False):
            time_sample_s = expected_time_s
            time_sample_prob = 1.0
        else:
            time_sample = sample_from_logits(
                top_t_logits,
                temperature=T_time,
                top_p=target_time_p,
                rng=rng,
            )
            time_sample_bin = int(time_sample.move_index)
            time_sample_s = _time_bin_to_seconds(time_sample_bin, active_clock_s, _time_budget_cap, num_time_bins)
            time_sample_prob = float(time_sample.prob)
    else:
        time_dist = []
        expected_time_s = 0.0
        mode_time_s = 0.0
        time_sample_s = 0.0
        time_sample_prob = 0.0

    top_policy_move = None
    resign_prob = 0.0
    flag_prob = 0.0
    resign_logit = float("-inf")
    flag_logit = float("-inf")
    # NOTE: kept as the field name for app/UCI compatibility; with dedicated
    # binary heads it now means "the resign/flag head probability cleared the
    # configured threshold", i.e. the engine should resign / flag.
    resign_beats_top_policy = False
    flag_beats_top_policy = False

    # Top legal move for display / logging.
    if legal_moves_data:
        legal_logits = np.array([x["logit"] for x in legal_moves_data], dtype=np.float64)
        legal_probs = _softmax_np(legal_logits.astype(np.float32))
        top_legal_idx = int(np.argmax(legal_logits))
        top_legal_entry = legal_moves_data[top_legal_idx]
        top_real_move = _canonical_to_real_move(top_legal_entry["move"], board.turn)
        try:
            top_label = board.san(top_real_move)
        except Exception:
            top_label = top_real_move.uci()
        top_policy_move = {
            "uci": top_real_move.uci(),
            "label": top_label,
            "prob": float(legal_probs[top_legal_idx]),
            "logit": float(top_legal_entry["logit"]),
        }

    # Resign / flag from their dedicated binary heads (sigmoid of the logit).
    # The position triggers resign/flag when the head probability clears the
    # configured threshold.
    if not skip_resign_flag:
        resign_logit = float(_as_numpy(resign_logit_t).reshape(-1)[0])
        resign_prob = float(1.0 / (1.0 + np.exp(-resign_logit)))
        resign_threshold = float(settings.get("resign_threshold", 0.98))
        resign_beats_top_policy = bool(resign_prob >= resign_threshold)
        if flag_logit_t is not None:
            flag_logit = float(_as_numpy(flag_logit_t).reshape(-1)[0])
            flag_prob = float(1.0 / (1.0 + np.exp(-flag_logit)))
            flag_threshold = float(settings.get("flag_threshold", 0.98))
            flag_beats_top_policy = bool(flag_prob >= flag_threshold)

    # Derive scalar value from WDL: win_prob = W + 0.5 * D
    win_prob = float(wdl[2] + 0.5 * wdl[1])

    return {
        "top_moves": policy_display,
        "resign": resign_prob,
        "flag": flag_prob,
        "resign_logit": float(resign_logit) if np.isfinite(resign_logit) else None,
        "flag_logit": float(flag_logit) if np.isfinite(flag_logit) else None,
        "top_policy_move": top_policy_move,
        "resign_beats_top_policy": resign_beats_top_policy,
        "flag_beats_top_policy": flag_beats_top_policy,
        "wdl": {"w": float(wdl[2]), "d": float(wdl[1]), "l": float(wdl[0])},
        "value": win_prob,
        "stm": "black" if board.turn == chess.BLACK else "white",
        "time_dist": time_dist,
        "time_top_p": target_time_p,
        "time_sample_s": float(time_sample_s),
        "time_sample_prob": float(time_sample_prob),
        "expected_time_s": float(expected_time_s),
        "mode_time_s": float(mode_time_s),
    }


def _effective_think_time_s(pred_s: float, settings: dict, active_clock_s: float) -> float:
    """Turn the model's raw predicted move time into the simulated think time.

    The model already conditions on the time control (base time, increment,
    remaining clocks), so its raw prediction is the honest think time. Two
    optional knobs adjust it:

      ``time_scale`` – multiplicative dial on the predicted time. 1.0 is a no-op
        (predicted == slept). Not a correctness fix; use it only to make the bot
        deliberately faster (<1) or slower (>1) than its own estimate. Being a
        linear multiply, it preserves the sampled distribution's variance /
        move-dependence (unlike a flat floor): 1s -> scale s, 2s -> 2*scale s.

      ``min_move_time_s`` – a small floor to avoid literal 0.0s replies. Kept
        intentionally low; a large floor flattens every move to a constant and
        looks robotic. The floor never consumes more than
        ``min_move_time_clock_fraction`` of the remaining clock so it can't
        push the bot into flagging itself.
    """
    scale = float(settings.get("time_scale", 1.0))
    t = max(0.0, pred_s) * scale

    min_s = float(settings.get("min_move_time_s", 0.0))
    if min_s > 0.0 and t < min_s:
        frac = float(settings.get("min_move_time_clock_fraction", 0.3))
        safe_cap = max(0.0, active_clock_s * frac)
        t = max(t, min(min_s, safe_cap))
    return t


def choose_engine_move(
    *,
    backend: "InferenceBackend",
    settings: dict,
    rng: np.random.Generator,
    board: chess.Board,
    moves_uci: list[str],
    active_clock_s: float,
    opponent_clock_s: float,
    active_inc_s: float = 0.0,
    opponent_inc_s: float = 0.0,
    time_history_s: list[float] | None,
    stop_check: Callable[[], bool] | None = None,
    allow_ponder_sleep: bool = True,
    mcts_progress_callback: Callable[[dict], None] | None = None,
    mcts_reuse_root: _Node | None = None,
    mcts_reuse_moves: list[chess.Move] | None = None,
    initial_fen: str = chess.STARTING_FEN,
) -> tuple[PolicyOutput, dict, dict | None, MCTSResult | None]:
    """Implements the engine selection path.

    Returns: (policy_out, analysis_dict, mcts_stats_or_none, mcts_result_or_none)
    
    The mcts_result contains the search tree root for potential reuse in subsequent searches.
    Pass mcts_reuse_root and mcts_reuse_moves to enable tree reuse from a previous search.
    """

    if board.is_game_over():
        return PolicyOutput(move=None, policy_prob=1.0), {}, None, None

    # UCI path: skip per-move value evaluations (web UI only) and skip the
    # resign/flag forward pass when both behaviours are disabled.
    _needs_resign_flag = (
        bool(settings.get("enable_resign", False))
        or bool(settings.get("enable_flag", False))
        or bool(settings.get("log_resign_probs", False))
    )
    engine_stats = analyze_position(
        backend=backend,
        settings=settings,
        rng=rng,
        board=board,
        moves_uci=moves_uci,
        active_clock_s=float(active_clock_s),
        opponent_clock_s=float(opponent_clock_s),
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        time_history_s=time_history_s,
        tc_base_s=(float(settings.get("start_clock_s")) if settings.get("start_clock_s") is not None else None),
        initial_fen=initial_fen,
        skip_move_values=True,
        skip_resign_flag=not _needs_resign_flag,
    )

    engine_pred_time_s = float(engine_stats.get("time_sample_s", 0.0))

    # Policy-weighted average time for adaptive budget when KL-regularized.
    # Instead of using only the top move's time, weight each move's expected
    # time by its policy probability for a more representative estimate.
    if bool(settings.get("mcts_kl_regularized", False)):
        top_moves = engine_stats.get("top_moves", [])
        if top_moves:
            total_prob = sum(e.get("prob", 0.0) for e in top_moves)
            if total_prob > 0:
                engine_pred_time_s = sum(
                    e.get("prob", 0.0) * e.get("expected_time_s", 0.0)
                    for e in top_moves
                ) / total_prob

    # For non-MCTS mode: simulate thinking time before returning
    # For MCTS mode: we handle timing after MCTS completes (see below)
    use_mcts = bool(settings.get("use_mcts", False))
    current_ply = len(moves_uci)
    mcts_start_ply = int(settings.get("mcts_start_ply", 0))

    # Skip MCTS for early plies to allow varied openings
    use_mcts_this_move = use_mcts and (current_ply >= mcts_start_ply)

    if allow_ponder_sleep and bool(settings.get("simulate_thinking_time", False)) and not use_mcts_this_move:
        # In the web app, this is an intentional delay. In UCI, we want to be
        # responsive to `stop`/new `position`, so make it interruptible.
        sleep_time_s = _effective_think_time_s(engine_pred_time_s, settings, active_clock_s)
        if stop_check is None:
            time.sleep(sleep_time_s)
        else:
            deadline = time.time() + max(0.0, sleep_time_s)
            while time.time() < deadline:
                if stop_check():
                    break
                time.sleep(0.05)

    ctx = ContextOptions(
        active_elo=_get_bot_elo(settings),
        opponent_elo=_get_opponent_elo(settings),
        active_clock_s=float(active_clock_s),
        opponent_clock_s=float(opponent_clock_s),
        active_inc_s=float(active_inc_s),
        opponent_inc_s=float(opponent_inc_s),
        tc_base_s=(float(settings.get("start_clock_s")) if settings.get("start_clock_s") is not None else None),
        halfmove_clock=int(board.halfmove_clock),
    )

    mcts_stats = None

    # Check if we should use MCTS (enabled and past start ply)
    if use_mcts_this_move:
        sims = int(settings.get("mcts_simulations", 128))
        cpuct = float(settings.get("mcts_c_puct", 1.5))

        if bool(settings.get("mcts_adaptive", False)):
            scale = float(settings.get("mcts_adaptive_scale", 100.0))
            # Adaptive sims: scale * predicted_time
            # No minimum clamp: allow sims to be 0 if scale*pred_time is 0
            sims = int(scale * engine_pred_time_s)

        mcts_settings = MCTSSettings(
            simulations=sims,
            c_puct=cpuct,
            cpuct_base=float(settings.get("mcts_cpuct_base", 19768)),
            cpuct_factor=float(settings.get("mcts_cpuct_factor", 2.0)),
            max_children=int(settings.get("mcts_max_children", 48)),
            max_depth=int(settings.get("mcts_max_depth", 96)),
            root_dirichlet_alpha=float(settings.get("mcts_root_dirichlet_alpha", 0.0)),
            root_exploration_frac=float(settings.get("mcts_root_exploration_frac", 0.0)),
            final_temperature=float(settings.get("mcts_final_temperature", 0.0)),
            final_top_p=float(settings.get("mcts_final_top_p", 0.95)),
            fpu_reduction=float(settings.get("mcts_fpu_reduction", 0.33)),
            contempt=float(settings.get("mcts_contempt", 0.15)),
            batch_size=int(settings.get("mcts_batch_size", 16)),
            tree_reuse=bool(settings.get("mcts_tree_reuse", False)),
            kl_regularized=bool(settings.get("mcts_kl_regularized", False)),
            kl_coeff=float(settings.get("mcts_kl_coeff", 1.0)),
        )

        # Track MCTS execution time for timing simulation
        mcts_start_time = time.time()

        # Try to find a reusable subtree from the previous search
        reuse_root: _Node | None = None
        if mcts_settings.tree_reuse and mcts_reuse_root is not None and mcts_reuse_moves:
            reuse_root = find_subtree_by_move_sequence(mcts_reuse_root, mcts_reuse_moves)

        mcts_result = mcts_choose_move(
            backend=backend,
            board=board,
            ctx=ctx,
            time_history_s=time_history_s,
            settings=mcts_settings,
            rng=rng,
            stop_check=stop_check,
            claim_draw=False,
            progress_callback=mcts_progress_callback,
            reuse_root=reuse_root,
        )
        out = mcts_result.output
        mcts_stats = out.stats

        mcts_elapsed_s = time.time() - mcts_start_time

        # Simulate remaining thinking time if:
        #   - UCI: MCTSSimulateTime is explicitly set, OR
        #   - Web app: simulate_thinking_time is on (allow_ponder_sleep gates web-app calls)
        should_sim_time = bool(settings.get("mcts_simulate_time", False)) or (
            allow_ponder_sleep and bool(settings.get("simulate_thinking_time", False))
        )
        if should_sim_time:
            target_time_s = _effective_think_time_s(engine_pred_time_s, settings, active_clock_s)
            remaining_time = target_time_s - mcts_elapsed_s
            if remaining_time > 0:
                if stop_check is None:
                    time.sleep(remaining_time)
                else:
                    deadline = time.time() + remaining_time
                    while time.time() < deadline:
                        if stop_check():
                            break
                        time.sleep(0.05)

        return out, engine_stats, mcts_stats, mcts_result
    else:
        sampled_move = _sample_effective_policy_move(
            board=board,
            top_moves=engine_stats.get("top_moves", []),
            rng=rng,
        )

        if sampled_move is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                out = PolicyOutput(move=None, policy_prob=1.0)
            else:
                out = PolicyOutput(move=legal_moves[0], policy_prob=1.0 / len(legal_moves))
        else:
            out = PolicyOutput(move=sampled_move.move, policy_prob=sampled_move.prob)

    return out, engine_stats, mcts_stats, None
