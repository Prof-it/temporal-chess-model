"""Per-game position extraction. Used by worker processes.

Lifts board encoding / time bucketing / quota machinery from process_pgn_balanced.py
verbatim, then layers the new per-(elo_bin × tc_cat) per-month cap and
in_balance_range tagging on top.

Differences from process_pgn_balanced.process_game:
  * Drops `balanced_keep` (no inline ELO filter — out-of-range rows flow through).
    * Drops the `T < 180` ultra-bullet/bullet filter; uses `T < 30` as a
        separate capped ultraBullet category.
  * `tc_cat` now uses Lichess perfType encoding (config.get_tc_category).
  * Adds `in_balance_range` (bool) to every row.
    * Caps each balanced (elo_bin, tc_cat) bucket per worker and also applies
        small global caps to extra time controls plus out-of-range rows.
  * Returns yields counter alongside rows so the orchestrator can aggregate.
"""
from __future__ import annotations

import io
import random
from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import chess
import chess.pgn

from .config import (
    GAME_ELO_MIN,
    HISTORY_LEN,
    elo_bin,
    get_tc_category,
)

# ─── Constants lifted verbatim from process_pgn_balanced.py ──────────────────
PIECE_MAP = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}
PROMOTION_MAP = {
    None: 0, chess.QUEEN: 1, chess.ROOK: 2,
    chess.BISHOP: 3, chess.KNIGHT: 4,
}

RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
RESIGN_SENTINEL_MOVE = [RESIGN_MOVE_INDEX, RESIGN_MOVE_INDEX, 0]
FLAG_SENTINEL_MOVE = [FLAG_MOVE_INDEX, FLAG_MOVE_INDEX, 0]

PLY_BANDS = [20, 40, 60, 80]
CLOCK_BANDS = [15, 60, 300, 600]
BUCKET_QUOTA = 150_000  # within-cell diversity cap (not the new (elo,tc) cap)
BAL_ELO_STEP_INTERNAL = 100  # for bucket_key elo banding (matches old code)

FLAG_TERMS = ("time forfeit", "forfeit on time", "flagged", "flag",
              "time ran out", "time expired")
RESIGN_TERMINATIONS = {"", "Normal"}
FLAG_TERMINATIONS = {"Time forfeit"}

# Output schema column order (matches what staging.py writes).
SCHEMA_COLUMNS = [
    "game_id", "split", "board_history", "legal_moves", "time_history",
    "active_elo", "opp_elo", "move_ply", "halfmove_clock", "repetition_flags",
    "castling_rights", "ep_square", "tc_cat", "base_time",
    "active_clock", "opp_clock", "active_inc", "opp_inc",
    "y_val", "policy_move", "policy_is_resign", "policy_is_flag", "time_target",
    "in_balance_range",
]

# ─── Worker-local state ──────────────────────────────────────────────────────
_w_quota_counts: Dict[Tuple[int, int, int], int] = {}
_w_quota_max: int = 0
_w_rng: random.Random = random.Random()
_w_id: int = 0
_w_caps: Dict[Tuple[int, int], int] = {}  # (elo_bin, tc_cat) → remaining cap
_w_yields: Dict[Tuple[int, int], int] = {}  # cumulative kept counts per bucket
_w_extra_tc_caps: Dict[int, int] = {}  # tc_cat → remaining global cap
_w_extra_tc_yields: Dict[int, int] = {}
_w_oor_cap: int = 0  # 0 = uncapped
_w_oor_yield: int = 0
_w_tc_filter: Optional[int] = None  # None = accept all TCs; else only this tc_cat


def init_worker(
    worker_id: int,
    bucket_quota: int,
    seed: int,
    per_worker_cap: Dict[Tuple[int, int], int],
    extra_tc_caps_per_worker: Optional[Dict[int, int]] = None,
    oor_cap_per_worker: int = 0,
    tc_filter: Optional[int] = None,
) -> None:
    """Initialize worker-local state. Called once per worker process startup.

    oor_cap_per_worker: 0 = uncapped; otherwise the per-worker OOR row budget
    for this month. After this many OOR rows the worker rejects further OOR.

    tc_filter: if set, only games with this tc_cat are processed. Others are
    skipped at header-parse time (cheap). Used for staged single-TC builds.
    """
    global _w_quota_counts, _w_quota_max, _w_rng, _w_id, _w_caps, _w_yields
    global _w_extra_tc_caps, _w_extra_tc_yields
    global _w_oor_cap, _w_oor_yield, _w_tc_filter
    _w_quota_counts = defaultdict(int)
    _w_quota_max = bucket_quota
    _w_rng = random.Random(seed + worker_id)
    _w_id = worker_id
    _w_caps = dict(per_worker_cap)
    _w_yields = defaultdict(int)
    _w_extra_tc_caps = dict(extra_tc_caps_per_worker or {})
    _w_extra_tc_yields = defaultdict(int)
    _w_oor_cap = max(0, int(oor_cap_per_worker))
    _w_oor_yield = 0
    _w_tc_filter = tc_filter


def get_yields() -> Dict[Tuple[int, int], int]:
    """Return a copy of cumulative per-bucket kept counts for this worker."""
    return dict(_w_yields)


def reset_yields() -> None:
    """Reset cumulative yields (used between months)."""
    global _w_yields
    _w_yields = defaultdict(int)


# ─── Lifted helpers ──────────────────────────────────────────────────────────
def parse_time_control(tc_string: str) -> Optional[Tuple[int, int]]:
    if not tc_string or tc_string == "-":
        return None
    try:
        if "+" in tc_string:
            base, inc = tc_string.split("+")
            return int(base), int(inc)
        return int(tc_string), 0
    except ValueError:
        return None


def encode_board(board: chess.Board) -> List[int]:
    tokens = [0] * 64
    for square in range(64):
        piece = board.piece_at(square)
        if not piece:
            continue
        val = PIECE_MAP[piece.piece_type]
        tokens[square] = val if piece.color == chess.WHITE else val + 6
    return tokens


def canonicalize(board: chess.Board, turn: chess.Color) -> chess.Board:
    return board if turn == chess.WHITE else board.mirror()


def encode_legal_moves(board: chess.Board) -> List[List[int]]:
    moves = []
    for move in board.legal_moves:
        promo = PROMOTION_MAP.get(move.promotion, 0)
        moves.append([move.from_square, move.to_square, promo])
    return moves


def repetition_flags(history: List[chess.Board]) -> List[int]:
    flags: List[int] = []
    seen: Counter = Counter()
    for b in history:
        key = b.board_fen()
        seen[key] += 1
        flags.append(1 if seen[key] >= 2 else 0)
    while len(flags) < HISTORY_LEN:
        flags.insert(0, 0)
    return flags[:HISTORY_LEN]


def bucket_key(elo: int, ply: int, clock_seconds: float,
               inc_seconds: float) -> Tuple[int, int, int, int]:
    if elo < GAME_ELO_MIN:
        b_elo = 0
    elif elo >= 2900:
        b_elo = (2900 - GAME_ELO_MIN) // BAL_ELO_STEP_INTERNAL + 1
    else:
        b_elo = (elo - GAME_ELO_MIN) // BAL_ELO_STEP_INTERNAL + 1
    b_ply = len(PLY_BANDS)
    for i, limit in enumerate(PLY_BANDS):
        if ply < limit:
            b_ply = i
            break
    b_clk = len(CLOCK_BANDS)
    for i, limit in enumerate(CLOCK_BANDS):
        if clock_seconds < limit:
            b_clk = i
            break
    b_inc = 1 if inc_seconds > 0 else 0
    return b_elo, b_ply, b_clk, b_inc


def quota_accept(bucket: Tuple[int, int, int, int]) -> bool:
    if _w_quota_max <= 0:
        return True
    key = bucket[:3]
    count = _w_quota_counts[key]
    if count >= _w_quota_max:
        return False
    _w_quota_counts[key] = count + 1
    return True


def cap_accept(b_elo: int, tc_cat: int) -> bool:
    """Per-row cap check for balanced buckets and capped extras.

    OOR (b_elo<0) is capped via _w_oor_cap if set. Classical and ultraBullet
    use their own small global caps, independent of the core balanced buckets.
    """
    if tc_cat in _w_extra_tc_caps:
        if _w_extra_tc_yields[tc_cat] >= _w_extra_tc_caps[tc_cat]:
            return False
    if b_elo < 0:
        # Out-of-range ELO — capped at _w_oor_cap (combined across all OOR bins/TCs).
        global _w_oor_yield
        if _w_oor_cap > 0 and _w_oor_yield >= _w_oor_cap:
            return False
        _w_oor_yield += 1
        if tc_cat in _w_extra_tc_caps:
            _w_extra_tc_yields[tc_cat] += 1
        return True
    if tc_cat in _w_extra_tc_caps:
        _w_extra_tc_yields[tc_cat] += 1
        return True
    cap = _w_caps.get((b_elo, tc_cat))
    if cap is None:
        # No cap registered for this bucket → uncapped.
        _w_yields[(b_elo, tc_cat)] += 1
        return True
    if _w_yields[(b_elo, tc_cat)] < cap:
        _w_yields[(b_elo, tc_cat)] += 1
        return True
    return False


def bucket_has_space(b_elo: int, tc_cat: int) -> bool:
    """Cheap, read-only version of cap_accept used for pre-skip.

    Returns True if the bucket can still accept rows (or is uncapped). Does NOT
    mutate yields/counts — for that, call cap_accept inside the parse loop.
    """
    if tc_cat in _w_extra_tc_caps and _w_extra_tc_yields[tc_cat] >= _w_extra_tc_caps[tc_cat]:
        return False
    if b_elo < 0:
        # OOR: check the global OOR cap.
        if _w_oor_cap <= 0:
            return True
        return _w_oor_yield < _w_oor_cap
    cap = _w_caps.get((b_elo, tc_cat))
    if cap is None:
        return True  # uncapped (classical)
    return _w_yields[(b_elo, tc_cat)] < cap


def extract_time_seconds(node: chess.pgn.ChildNode) -> Optional[float]:
    clk = node.clock()
    if clk is None:
        return None
    try:
        return float(clk)
    except ValueError:
        return None


# ─── Main per-game extractor ─────────────────────────────────────────────────
def process_game(game: chess.pgn.Game) -> List[dict]:
    """Return list of row dicts for a single PGN game.

    Worker-local quotas (bucket diversity + per-(elo_bin, tc_cat) cap) are
    consulted inline; rows that don't fit the cap are dropped.
    """
    headers = game.headers
    tc = parse_time_control(headers.get("TimeControl", ""))
    if not tc:
        return []
    base, inc = tc
    tc_cat = get_tc_category(base, inc)
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
    except ValueError:
        return []
    if white_elo < GAME_ELO_MIN or black_elo < GAME_ELO_MIN:
        return []

    result = headers.get("Result", "*")
    termination = headers.get("Termination", "")
    site = headers.get("Site", "unknown")
    game_id = site.rsplit("/", 1)[-1] if "/" in site else site
    is_flag_loss = any(term in termination.lower() for term in FLAG_TERMS)

    board = game.board()
    node = game
    ply = 0
    board_history: Deque[List[int]] = deque(maxlen=HISTORY_LEN)
    raw_boards: Deque[chess.Board] = deque(maxlen=HISTORY_LEN)
    time_history: Deque[float] = deque(maxlen=HISTORY_LEN)
    rows: List[dict] = []

    white_clock = float(base)
    black_clock = float(base)
    white_last = 0.0
    black_last = 0.0

    while True:
        next_node = node.next()
        if next_node is None:
            break
        move = next_node.move
        turn = board.turn

        canonical = canonicalize(board, turn)
        board_tokens = encode_board(canonical)
        board_history.appendleft(board_tokens)
        raw_boards.appendleft(canonical.copy())

        turn_clock = white_clock if turn == chess.WHITE else black_clock
        opp_clock = black_clock if turn == chess.WHITE else white_clock
        turn_inc = inc
        opp_inc = inc

        if turn == chess.WHITE:
            active_elo, opp_elo_val = white_elo, black_elo
            policy_from = move.from_square
            policy_to = move.to_square
            policy_promo = PROMOTION_MAP.get(move.promotion, 0)
            y_val = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5
        else:
            active_elo, opp_elo_val = black_elo, white_elo
            policy_from = move.from_square ^ 56
            policy_to = move.to_square ^ 56
            policy_promo = PROMOTION_MAP.get(move.promotion, 0)
            y_val = 0.0 if result == "1-0" else 1.0 if result == "0-1" else 0.5

        bucket = bucket_key(active_elo, ply, turn_clock, turn_inc)
        keep_diversity = quota_accept(bucket)

        row_ref: Optional[dict] = None

        if keep_diversity:
            b_elo = elo_bin(active_elo)
            in_range = b_elo >= 0
            if cap_accept(b_elo, tc_cat):
                history_tensor = list(board_history)
                while len(history_tensor) < HISTORY_LEN:
                    history_tensor.append([0] * 64)
                history_tensor = history_tensor[:HISTORY_LEN]

                rep = repetition_flags(list(raw_boards))
                legal_moves = encode_legal_moves(canonical)

                time_hist = list(time_history)
                while len(time_hist) < HISTORY_LEN:
                    time_hist.insert(0, 0.0)
                time_hist = time_hist[-HISTORY_LEN:]

                pos_id = f"{game_id}_{ply}"
                row = {
                    "game_id": pos_id,
                    "split": "train",
                    "board_history": history_tensor,
                    "legal_moves": legal_moves,
                    "time_history": time_hist,
                    "active_elo": active_elo,
                    "opp_elo": opp_elo_val,
                    "move_ply": ply,
                    "halfmove_clock": board.halfmove_clock,
                    "repetition_flags": rep,
                    "castling_rights": [
                        int(board.has_kingside_castling_rights(chess.WHITE)),
                        int(board.has_queenside_castling_rights(chess.WHITE)),
                        int(board.has_kingside_castling_rights(chess.BLACK)),
                        int(board.has_queenside_castling_rights(chess.BLACK)),
                    ],
                    "ep_square": board.ep_square if board.ep_square is not None else -1,
                    "tc_cat": tc_cat,
                    "base_time": base,
                    "active_clock": turn_clock,
                    "opp_clock": opp_clock,
                    "active_inc": turn_inc,
                    "opp_inc": opp_inc,
                    "y_val": y_val,
                    "policy_move": [policy_from, policy_to, policy_promo],
                    "policy_is_resign": False,
                    "policy_is_flag": False,
                    "time_target": 0.0,
                    "in_balance_range": in_range,
                }
                rows.append(row)
                row_ref = row

        # Update clocks
        clk_seconds = extract_time_seconds(next_node)
        if clk_seconds is not None:
            elapsed = max(0.0, turn_clock + turn_inc - clk_seconds)
            time_history.append(elapsed)
            if turn == chess.WHITE:
                white_clock = clk_seconds
                white_last = elapsed
            else:
                black_clock = clk_seconds
                black_last = elapsed
            if row_ref is not None:
                row_ref["time_target"] = elapsed
        else:
            time_history.append(0.0)

        board.push(move)
        ply += 1
        node = next_node

    # ── Resign positions ──
    if result in {"1-0", "0-1"} and termination in RESIGN_TERMINATIONS:
        if not board.is_checkmate() and not board.is_stalemate():
            loser = chess.BLACK if result == "1-0" else chess.WHITE
            if board.turn == loser:
                canonical = canonicalize(board, loser)
                board_history.appendleft(encode_board(canonical))
                raw_boards.appendleft(canonical.copy())

                active_elo = white_elo if loser == chess.WHITE else black_elo
                opp_elo_val = black_elo if loser == chess.WHITE else white_elo
                active_clock = white_clock if loser == chess.WHITE else black_clock

                bucket = bucket_key(active_elo, ply, active_clock, inc)
                b_elo = elo_bin(active_elo)
                in_range = b_elo >= 0
                pos_id = f"{game_id}_{ply}_resign"
                if quota_accept(bucket) and cap_accept(b_elo, tc_cat):
                    history_tensor = list(board_history)
                    while len(history_tensor) < HISTORY_LEN:
                        history_tensor.append([0] * 64)
                    time_hist = list(time_history)
                    while len(time_hist) < HISTORY_LEN:
                        time_hist.insert(0, 0.0)
                    rep = repetition_flags(list(raw_boards))
                    legal_moves = encode_legal_moves(canonical)
                    rows.append({
                        "game_id": pos_id,
                        "split": "train",
                        "board_history": history_tensor[:HISTORY_LEN],
                        "legal_moves": legal_moves,
                        "time_history": time_hist[-HISTORY_LEN:],
                        "active_elo": active_elo,
                        "opp_elo": opp_elo_val,
                        "move_ply": ply,
                        "halfmove_clock": board.halfmove_clock,
                        "repetition_flags": rep,
                        "castling_rights": [
                            int(board.has_kingside_castling_rights(chess.WHITE)),
                            int(board.has_queenside_castling_rights(chess.WHITE)),
                            int(board.has_kingside_castling_rights(chess.BLACK)),
                            int(board.has_queenside_castling_rights(chess.BLACK)),
                        ],
                        "ep_square": board.ep_square if board.ep_square is not None else -1,
                        "tc_cat": tc_cat,
                        "base_time": base,
                        "active_clock": active_clock,
                        "opp_clock": black_clock if loser == chess.WHITE else white_clock,
                        "active_inc": inc,
                        "opp_inc": inc,
                        "y_val": 0.0,
                        "policy_move": RESIGN_SENTINEL_MOVE,
                        "policy_is_resign": True,
                        "policy_is_flag": False,
                        "time_target": 3.0,
                        "in_balance_range": in_range,
                    })

    # ── Flag loss positions ──
    if (is_flag_loss or termination in FLAG_TERMINATIONS) and result in {"1-0", "0-1"}:
        loser = chess.BLACK if result == "1-0" else chess.WHITE
        if board.turn == loser:
            canonical = canonicalize(board, loser)
            board_history.appendleft(encode_board(canonical))
            raw_boards.appendleft(canonical.copy())

            active_elo = white_elo if loser == chess.WHITE else black_elo
            opp_elo_val = black_elo if loser == chess.WHITE else white_elo
            active_clock = white_clock if loser == chess.WHITE else black_clock

            bucket = bucket_key(active_elo, ply, active_clock, inc)
            b_elo = elo_bin(active_elo)
            in_range = b_elo >= 0
            pos_id = f"{game_id}_{ply}_flag"
            if quota_accept(bucket) and cap_accept(b_elo, tc_cat):
                history_tensor = list(board_history)
                while len(history_tensor) < HISTORY_LEN:
                    history_tensor.append([0] * 64)
                time_hist = list(time_history)
                while len(time_hist) < HISTORY_LEN:
                    time_hist.insert(0, 0.0)
                rep = repetition_flags(list(raw_boards))
                legal_moves = encode_legal_moves(canonical)
                flag_time_target = white_last if loser == chess.WHITE else black_last
                rows.append({
                    "game_id": pos_id,
                    "split": "train",
                    "board_history": history_tensor[:HISTORY_LEN],
                    "legal_moves": legal_moves,
                    "time_history": time_hist[-HISTORY_LEN:],
                    "active_elo": active_elo,
                    "opp_elo": opp_elo_val,
                    "move_ply": ply,
                    "halfmove_clock": board.halfmove_clock,
                    "repetition_flags": rep,
                    "castling_rights": [
                        int(board.has_kingside_castling_rights(chess.WHITE)),
                        int(board.has_queenside_castling_rights(chess.WHITE)),
                        int(board.has_kingside_castling_rights(chess.BLACK)),
                        int(board.has_queenside_castling_rights(chess.BLACK)),
                    ],
                    "ep_square": board.ep_square if board.ep_square is not None else -1,
                    "tc_cat": tc_cat,
                    "base_time": base,
                    "active_clock": active_clock,
                    "opp_clock": black_clock if loser == chess.WHITE else white_clock,
                    "active_inc": inc,
                    "opp_inc": inc,
                    "y_val": 0.0,
                    "policy_move": FLAG_SENTINEL_MOVE,
                    "policy_is_resign": False,
                    "policy_is_flag": True,
                    "time_target": flag_time_target,
                    "in_balance_range": in_range,
                })

    return rows


def process_game_text(text: str) -> List[dict]:
    """Parse a PGN game text block and extract rows. Returns [] on parse error.

    Pre-skip optimization: reads only the headers first to determine the game's
    (elo_bin, tc_cat). If both white's and black's buckets are already full,
    skips full move parsing entirely — saves ~95% of CPU on filled buckets.
    """
    try:
        buf = io.StringIO(text)
        headers = chess.pgn.read_headers(buf)
        if headers is None:
            return []

        # Cheap pre-checks — avoid full parse when we know we'd reject.
        tc = parse_time_control(headers.get("TimeControl", ""))
        if not tc:
            return []
        base, inc = tc
        tc_cat = get_tc_category(base, inc)
        # TC filter (staged builds): skip games not in the target TC.
        if _w_tc_filter is not None and tc_cat != _w_tc_filter:
            return []

        try:
            white_elo = int(headers.get("WhiteElo", 0))
            black_elo = int(headers.get("BlackElo", 0))
        except ValueError:
            return []
        if white_elo < GAME_ELO_MIN or black_elo < GAME_ELO_MIN:
            return []

        # If BOTH players' buckets are full, we'd extract nothing — skip parse.
        # Capped extras still benefit from the same pre-skip once their global
        # allowance is exhausted.
        from .config import elo_bin as _elo_bin
        b_white = _elo_bin(white_elo)
        b_black = _elo_bin(black_elo)
        if not bucket_has_space(b_white, tc_cat) and not bucket_has_space(b_black, tc_cat):
            return []

        # Worth a full parse. Re-read from the start.
        buf.seek(0)
        game = chess.pgn.read_game(buf)
        if game is None:
            return []
        return process_game(game)
    except Exception:
        return []
