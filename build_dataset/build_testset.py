#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterator, Optional

import chess
import chess.pgn
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from tqdm import tqdm


HISTORY_LEN = 8
CHUNK_SIZE = 250_000
DEFAULT_TARGET_POSITIONS = 5_000_000

RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
RESIGN_SENTINEL_MOVE = [RESIGN_MOVE_INDEX, RESIGN_MOVE_INDEX, 0]
FLAG_SENTINEL_MOVE = [FLAG_MOVE_INDEX, FLAG_MOVE_INDEX, 0]

ANALYSIS_TIME_BUCKETS = [0.05, 0.2, 0.5]
ELO_MIN_DEFAULT = 1200
ELO_MAX_DEFAULT = 2500

FLAG_TERMS = (
    "time forfeit",
    "forfeit on time",
    "flagged",
    "flag",
    "time ran out",
    "time expired",
)
RESIGN_TERMINATIONS = {"", "Normal"}
FLAG_TERMINATIONS = {"Time forfeit"}

PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}
PROMOTION_MAP = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}

SCHEMA_COLUMNS = [
    "game_id",
    "ply",
    "board_history",
    "legal_moves",
    "time_history",
    "active_elo",
    "opp_elo",
    "move_ply",
    "halfmove_clock",
    "repetition_flags",
    "castling_rights",
    "ep_square",
    "tc_cat",
    "tc_group",
    "tc_base_s",
    "tc_inc_s",
    "active_clock",
    "opp_clock",
    "active_inc",
    "opp_inc",
    "y_val",
    "policy_move",
    "policy_is_resign",
    "policy_is_flag",
    "time_target",
    "game_total_plies",
    "plies_to_end",
]


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_dir: Path
    mode: str
    target_positions: int
    min_active_elo: int
    max_active_elo: int
    tc_group: str
    selection_mode: str
    seed: int
    chunk_size: int
    overwrite: bool


@dataclass(frozen=True)
class Plan:
    mode: str
    selection_mode: str
    target_positions: int
    eligible_rows: int
    keep_probability: float
    cell_keep_probabilities: dict[tuple[int, int], float]
    cell_counts: dict[tuple[int, int], int]
    cell_target: int


def parse_time_control(tc_string: str) -> Optional[tuple[float, float]]:
    if not tc_string or tc_string == "-":
        return None
    try:
        if "+" in tc_string:
            base, inc = tc_string.split("+", 1)
            return float(base), float(inc)
        return float(tc_string), 0.0
    except ValueError:
        return None


def tc_group_from_duration(duration_s: float) -> str:
    if duration_s < 180:
        return "bullet"
    if duration_s < 600:
        return "blitz"
    if duration_s < 1800:
        return "rapid"
    return "classical"


def legacy_tc_cat(duration_s: float) -> int:
    if duration_s < 600:
        return 0
    if duration_s < 1800:
        return 1
    return 2


def encode_board(board: chess.Board) -> list[int]:
    tokens = [0] * 64
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue
        value = PIECE_MAP[piece.piece_type]
        tokens[square] = value if piece.color == chess.WHITE else value + 6
    return tokens


def canonicalize(board: chess.Board, turn: chess.Color) -> chess.Board:
    return board if turn == chess.WHITE else board.mirror()


def encode_legal_moves(board: chess.Board) -> list[list[int]]:
    moves: list[list[int]] = []
    for move in board.legal_moves:
        moves.append([move.from_square, move.to_square, PROMOTION_MAP.get(move.promotion, 0)])
    return moves


def repetition_flags(history: list[chess.Board]) -> list[int]:
    flags: list[int] = []
    seen: Counter[str] = Counter()
    for board in history:
        key = board.board_fen()
        seen[key] += 1
        flags.append(1 if seen[key] >= 2 else 0)
    while len(flags) < HISTORY_LEN:
        flags.insert(0, 0)
    return flags[:HISTORY_LEN]


def extract_time_seconds(node: chess.pgn.ChildNode) -> Optional[float]:
    clk = node.clock()
    if clk is None:
        return None
    try:
        return float(clk)
    except ValueError:
        return None


def active_elo_in_range(active_elo: int, config: Config) -> bool:
    return config.min_active_elo <= active_elo < config.max_active_elo


def clipped_clock_ratio(active_clock: float, tc_base_s: float) -> float:
    return min(max(float(active_clock) / max(float(tc_base_s), 1.0), 0.0), 1.0)


def analysis_time_bucket(active_clock: float, tc_base_s: float) -> int:
    ratio = clipped_clock_ratio(active_clock, tc_base_s)
    if ratio <= ANALYSIS_TIME_BUCKETS[0]:
        return 0
    if ratio <= ANALYSIS_TIME_BUCKETS[1]:
        return 1
    if ratio <= ANALYSIS_TIME_BUCKETS[2]:
        return 2
    return 3


def elo_bin(active_elo: int, config: Config) -> Optional[int]:
    if not active_elo_in_range(active_elo, config):
        return None
    return (active_elo - config.min_active_elo) // 100


def analysis_cell(active_elo: int, active_clock: float, tc_base_s: float, config: Config) -> Optional[tuple[int, int]]:
    elo_bucket = elo_bin(active_elo, config)
    if elo_bucket is None:
        return None
    return elo_bucket, analysis_time_bucket(active_clock, tc_base_s)


def analysis_total_cells(config: Config) -> int:
    elo_bins = max(1, (config.max_active_elo - config.min_active_elo) // 100)
    return elo_bins * 4


def deterministic_unit_interval(key: str, seed: int) -> float:
    raw = f"{key}|{seed}".encode("utf-8")
    digest = hashlib.blake2s(raw, digest_size=8).digest()
    return int.from_bytes(digest, "little") / (2 ** 64)


def keep_with_probability(key: str, probability: float, seed: int) -> bool:
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    return deterministic_unit_interval(key, seed) < probability


def ensure_output_dir(config: Config) -> None:
    if config.output_dir.exists() and any(config.output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(
            f"Output directory {config.output_dir} is not empty. Use --overwrite to reuse it."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _iter_games_plain(path: Path) -> Iterator[chess.pgn.Game]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            yield game


def _iter_games_zst(path: Path) -> Iterator[chess.pgn.Game]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as handle:
        reader = dctx.stream_reader(handle)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while True:
            try:
                game = chess.pgn.read_game(text_stream)
            except Exception:
                continue
            if game is None:
                break
            yield game


def iter_games(path: Path) -> Iterator[chess.pgn.Game]:
    if path.suffix == ".zst":
        return _iter_games_zst(path)
    return _iter_games_plain(path)


class ShardedParquetWriter:
    def __init__(self, output_dir: Path, prefix: str, chunk_size: int = CHUNK_SIZE):
        self.output_dir = output_dir
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.buffer: list[dict] = []
        self.shard_idx = 0
        self.total_written = 0

    def add_rows(self, rows: list[dict]) -> None:
        self.buffer.extend(rows)
        while len(self.buffer) >= self.chunk_size:
            self._flush(self.chunk_size)

    def finalize(self) -> int:
        if self.buffer:
            self._flush(len(self.buffer))
        return self.total_written

    def _flush(self, count: int) -> None:
        rows = self.buffer[:count]
        self.buffer = self.buffer[count:]
        dataframe = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
        table = pa.Table.from_pandas(dataframe, preserve_index=False)
        shard_path = self.output_dir / f"{self.prefix}_{self.shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        self.total_written += len(rows)
        self.shard_idx += 1
        del dataframe, table, rows
        gc.collect()


def parse_game_metadata(game: chess.pgn.Game, config: Config) -> Optional[dict[str, object]]:
    headers = game.headers
    tc = parse_time_control(headers.get("TimeControl", ""))
    if tc is None:
        return None
    tc_base_s, tc_inc_s = tc
    duration_s = tc_base_s + 40 * tc_inc_s
    tc_group = tc_group_from_duration(duration_s)
    if config.tc_group != "all" and tc_group != config.tc_group:
        return None

    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
    except ValueError:
        return None

    result = headers.get("Result", "*")
    termination = headers.get("Termination", "")
    game_id = headers.get("Site", "unknown").split("/")[-1]
    total_plies = sum(1 for _ in game.mainline_moves())

    return {
        "game_id": game_id,
        "tc_base_s": float(tc_base_s),
        "tc_inc_s": float(tc_inc_s),
        "tc_group": tc_group,
        "tc_cat": legacy_tc_cat(duration_s),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "termination": termination,
        "is_flag_loss": any(term in termination.lower() for term in FLAG_TERMS),
        "total_plies": total_plies,
    }


def scan_game(game: chess.pgn.Game, config: Config) -> tuple[int, Counter[tuple[int, int]]]:
    metadata = parse_game_metadata(game, config)
    if metadata is None:
        return 0, Counter()

    board = game.board()
    node = game
    ply = 0
    eligible_rows = 0
    cell_counts: Counter[tuple[int, int]] = Counter()

    white_clock = float(metadata["tc_base_s"])
    black_clock = float(metadata["tc_base_s"])

    while True:
        next_node = node.next()
        if next_node is None:
            break

        turn = board.turn
        active_elo = int(metadata["white_elo"]) if turn == chess.WHITE else int(metadata["black_elo"])
        active_clock = white_clock if turn == chess.WHITE else black_clock

        if active_elo_in_range(active_elo, config):
            eligible_rows += 1
            if config.mode == "analysis":
                cell = analysis_cell(active_elo, active_clock, float(metadata["tc_base_s"]), config)
                if cell is not None:
                    cell_counts[cell] += 1

        clk_seconds = extract_time_seconds(next_node)
        if clk_seconds is not None:
            if turn == chess.WHITE:
                white_clock = clk_seconds
            else:
                black_clock = clk_seconds

        board.push(next_node.move)
        ply += 1
        node = next_node

    result = str(metadata["result"])
    termination = str(metadata["termination"])
    loser = chess.BLACK if result == "1-0" else chess.WHITE if result == "0-1" else None
    if loser is not None and board.turn == loser:
        active_elo = int(metadata["white_elo"]) if loser == chess.WHITE else int(metadata["black_elo"])
        active_clock = white_clock if loser == chess.WHITE else black_clock
        if active_elo_in_range(active_elo, config):
            if result in {"1-0", "0-1"} and termination in RESIGN_TERMINATIONS and not board.is_checkmate() and not board.is_stalemate():
                eligible_rows += 1
                if config.mode == "analysis":
                    cell = analysis_cell(active_elo, active_clock, float(metadata["tc_base_s"]), config)
                    if cell is not None:
                        cell_counts[cell] += 1
            if result in {"1-0", "0-1"} and (bool(metadata["is_flag_loss"]) or termination in FLAG_TERMINATIONS):
                eligible_rows += 1
                if config.mode == "analysis":
                    cell = analysis_cell(active_elo, active_clock, float(metadata["tc_base_s"]), config)
                    if cell is not None:
                        cell_counts[cell] += 1

    return eligible_rows, cell_counts


def build_scan_plan(config: Config) -> Plan:
    eligible_rows = 0
    cell_counts: Counter[tuple[int, int]] = Counter()

    progress = tqdm(iter_games(config.input_path), desc="Pass 1/2: scan", unit="game")
    for game in progress:
        game_rows, game_cells = scan_game(game, config)
        eligible_rows += game_rows
        cell_counts.update(game_cells)
        progress.set_postfix(rows=eligible_rows)
    progress.close()

    if config.mode == "test":
        keep_probability = 1.0
        if config.target_positions > 0 and eligible_rows > config.target_positions:
            keep_probability = config.target_positions / eligible_rows
        return Plan(
            mode=config.mode,
            selection_mode=config.selection_mode,
            target_positions=config.target_positions,
            eligible_rows=eligible_rows,
            keep_probability=keep_probability,
            cell_keep_probabilities={},
            cell_counts={},
            cell_target=0,
        )

    non_empty_cells = {cell: count for cell, count in cell_counts.items() if count > 0}
    if not non_empty_cells:
        return Plan(
            mode=config.mode,
            selection_mode=config.selection_mode,
            target_positions=config.target_positions,
            eligible_rows=eligible_rows,
            keep_probability=0.0,
            cell_keep_probabilities={},
            cell_counts={},
            cell_target=0,
        )

    min_cell_count = min(non_empty_cells.values())
    target_per_cell = min_cell_count
    if config.target_positions > 0:
        target_per_cell = min(min_cell_count, max(1, config.target_positions // len(non_empty_cells)))

    cell_keep_probabilities = {
        cell: min(1.0, target_per_cell / count)
        for cell, count in non_empty_cells.items()
    }
    return Plan(
        mode=config.mode,
        selection_mode=config.selection_mode,
        target_positions=config.target_positions,
        eligible_rows=eligible_rows,
        keep_probability=0.0,
        cell_keep_probabilities=cell_keep_probabilities,
        cell_counts=dict(non_empty_cells),
        cell_target=target_per_cell,
    )


def build_stream_plan(config: Config) -> Plan:
    cell_target = 0
    if config.mode == "analysis" and config.target_positions > 0:
        cell_target = max(1, config.target_positions // analysis_total_cells(config))
    return Plan(
        mode=config.mode,
        selection_mode=config.selection_mode,
        target_positions=config.target_positions,
        eligible_rows=-1,
        keep_probability=1.0,
        cell_keep_probabilities={},
        cell_counts={},
        cell_target=cell_target,
    )


def build_plan(config: Config) -> Plan:
    if config.selection_mode == "scan":
        return build_scan_plan(config)
    return build_stream_plan(config)


def row_should_be_kept(
    row_id: str,
    active_elo: int,
    active_clock: float,
    tc_base_s: float,
    config: Config,
    plan: Plan,
    kept_cell_counts: Counter[tuple[int, int]],
) -> bool:
    if plan.mode == "test":
        if plan.selection_mode == "stream":
            return True
        return keep_with_probability(row_id, plan.keep_probability, config.seed)

    cell = analysis_cell(active_elo, active_clock, tc_base_s, config)
    if cell is None:
        return False
    if plan.selection_mode == "stream":
        if plan.cell_target <= 0:
            return True
        if kept_cell_counts[cell] >= plan.cell_target:
            return False
        kept_cell_counts[cell] += 1
        return True
    keep_probability = plan.cell_keep_probabilities.get(cell, 0.0)
    return keep_with_probability(row_id, keep_probability, config.seed)


def build_row(
    *,
    row_id: str,
    ply: int,
    board: chess.Board,
    turn: chess.Color,
    board_history: Deque[list[int]],
    raw_boards: Deque[chess.Board],
    time_history: Deque[float],
    active_elo: int,
    opp_elo: int,
    tc_cat: int,
    tc_group: str,
    tc_base_s: float,
    tc_inc_s: float,
    active_clock: float,
    opp_clock: float,
    active_inc: float,
    opp_inc: float,
    y_val: float,
    policy_move: list[int],
    policy_is_resign: bool,
    policy_is_flag: bool,
    time_target: float,
    game_total_plies: int,
) -> dict:
    canonical = canonicalize(board, turn)
    history_tensor = list(board_history)
    while len(history_tensor) < HISTORY_LEN:
        history_tensor.append([0] * 64)
    history_tensor = history_tensor[:HISTORY_LEN]

    time_hist = list(time_history)
    while len(time_hist) < HISTORY_LEN:
        time_hist.insert(0, 0.0)
    time_hist = time_hist[-HISTORY_LEN:]

    return {
        "game_id": row_id,
        "ply": ply,
        "board_history": history_tensor,
        "legal_moves": encode_legal_moves(canonical),
        "time_history": time_hist,
        "active_elo": active_elo,
        "opp_elo": opp_elo,
        "move_ply": ply,
        "halfmove_clock": board.halfmove_clock,
        "repetition_flags": repetition_flags(list(raw_boards)),
        "castling_rights": [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        ],
        "ep_square": board.ep_square if board.ep_square is not None else -1,
        "tc_cat": tc_cat,
        "tc_group": tc_group,
        "tc_base_s": tc_base_s,
        "tc_inc_s": tc_inc_s,
        "active_clock": active_clock,
        "opp_clock": opp_clock,
        "active_inc": active_inc,
        "opp_inc": opp_inc,
        "y_val": y_val,
        "policy_move": policy_move,
        "policy_is_resign": policy_is_resign,
        "policy_is_flag": policy_is_flag,
        "time_target": time_target,
        "game_total_plies": game_total_plies,
        "plies_to_end": max(0, game_total_plies - ply),
    }


def extract_rows_from_game(game: chess.pgn.Game, config: Config, plan: Plan, kept_cell_counts: Counter[tuple[int, int]]) -> list[dict]:
    metadata = parse_game_metadata(game, config)
    if metadata is None:
        return []

    board = game.board()
    node = game
    ply = 0
    rows: list[dict] = []

    board_history: Deque[list[int]] = deque(maxlen=HISTORY_LEN)
    raw_boards: Deque[chess.Board] = deque(maxlen=HISTORY_LEN)
    time_history: Deque[float] = deque(maxlen=HISTORY_LEN)

    white_clock = float(metadata["tc_base_s"])
    black_clock = float(metadata["tc_base_s"])
    white_last = 0.0
    black_last = 0.0

    while True:
        next_node = node.next()
        if next_node is None:
            break

        move = next_node.move
        turn = board.turn
        canonical = canonicalize(board, turn)
        board_history.appendleft(encode_board(canonical))
        raw_boards.appendleft(canonical.copy())

        turn_clock = white_clock if turn == chess.WHITE else black_clock
        opp_clock = black_clock if turn == chess.WHITE else white_clock
        active_inc = float(metadata["tc_inc_s"])
        opp_inc = float(metadata["tc_inc_s"])

        if turn == chess.WHITE:
            active_elo = int(metadata["white_elo"])
            opp_elo = int(metadata["black_elo"])
            policy_from = move.from_square
            policy_to = move.to_square
            y_val = 1.0 if metadata["result"] == "1-0" else 0.0 if metadata["result"] == "0-1" else 0.5
        else:
            active_elo = int(metadata["black_elo"])
            opp_elo = int(metadata["white_elo"])
            policy_from = move.from_square ^ 56
            policy_to = move.to_square ^ 56
            y_val = 0.0 if metadata["result"] == "1-0" else 1.0 if metadata["result"] == "0-1" else 0.5

        row_ref: Optional[dict] = None
        if active_elo_in_range(active_elo, config):
            row_id = f"{metadata['game_id']}_{ply}"
            if row_should_be_kept(row_id, active_elo, turn_clock, float(metadata["tc_base_s"]), config, plan, kept_cell_counts):
                row_ref = build_row(
                    row_id=row_id,
                    ply=ply,
                    board=board,
                    turn=turn,
                    board_history=board_history,
                    raw_boards=raw_boards,
                    time_history=time_history,
                    active_elo=active_elo,
                    opp_elo=opp_elo,
                    tc_cat=int(metadata["tc_cat"]),
                    tc_group=str(metadata["tc_group"]),
                    tc_base_s=float(metadata["tc_base_s"]),
                    tc_inc_s=float(metadata["tc_inc_s"]),
                    active_clock=turn_clock,
                    opp_clock=opp_clock,
                    active_inc=active_inc,
                    opp_inc=opp_inc,
                    y_val=y_val,
                    policy_move=[policy_from, policy_to, PROMOTION_MAP.get(move.promotion, 0)],
                    policy_is_resign=False,
                    policy_is_flag=False,
                    time_target=0.0,
                    game_total_plies=int(metadata["total_plies"]),
                )
                rows.append(row_ref)

        clk_seconds = extract_time_seconds(next_node)
        if clk_seconds is not None:
            elapsed = max(0.0, turn_clock + active_inc - clk_seconds)
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

    result = str(metadata["result"])
    termination = str(metadata["termination"])
    loser = chess.BLACK if result == "1-0" else chess.WHITE if result == "0-1" else None
    if loser is None or board.turn != loser:
        return rows

    canonical = canonicalize(board, loser)
    board_history.appendleft(encode_board(canonical))
    raw_boards.appendleft(canonical.copy())

    active_elo = int(metadata["white_elo"]) if loser == chess.WHITE else int(metadata["black_elo"])
    opp_elo = int(metadata["black_elo"]) if loser == chess.WHITE else int(metadata["white_elo"])
    active_clock = white_clock if loser == chess.WHITE else black_clock
    opp_clock = black_clock if loser == chess.WHITE else white_clock
    active_inc = float(metadata["tc_inc_s"])
    opp_inc = float(metadata["tc_inc_s"])

    if active_elo_in_range(active_elo, config):
        if result in {"1-0", "0-1"} and termination in RESIGN_TERMINATIONS and not board.is_checkmate() and not board.is_stalemate():
            row_id = f"{metadata['game_id']}_{ply}_resign"
            if row_should_be_kept(row_id, active_elo, active_clock, float(metadata["tc_base_s"]), config, plan, kept_cell_counts):
                rows.append(
                    build_row(
                        row_id=row_id,
                        ply=ply,
                        board=board,
                        turn=loser,
                        board_history=board_history,
                        raw_boards=raw_boards,
                        time_history=time_history,
                        active_elo=active_elo,
                        opp_elo=opp_elo,
                        tc_cat=int(metadata["tc_cat"]),
                        tc_group=str(metadata["tc_group"]),
                        tc_base_s=float(metadata["tc_base_s"]),
                        tc_inc_s=float(metadata["tc_inc_s"]),
                        active_clock=active_clock,
                        opp_clock=opp_clock,
                        active_inc=active_inc,
                        opp_inc=opp_inc,
                        y_val=0.0,
                        policy_move=RESIGN_SENTINEL_MOVE,
                        policy_is_resign=True,
                        policy_is_flag=False,
                        time_target=3.0,
                        game_total_plies=int(metadata["total_plies"]),
                    )
                )

        if result in {"1-0", "0-1"} and (bool(metadata["is_flag_loss"]) or termination in FLAG_TERMINATIONS):
            row_id = f"{metadata['game_id']}_{ply}_flag"
            if row_should_be_kept(row_id, active_elo, active_clock, float(metadata["tc_base_s"]), config, plan, kept_cell_counts):
                rows.append(
                    build_row(
                        row_id=row_id,
                        ply=ply,
                        board=board,
                        turn=loser,
                        board_history=board_history,
                        raw_boards=raw_boards,
                        time_history=time_history,
                        active_elo=active_elo,
                        opp_elo=opp_elo,
                        tc_cat=int(metadata["tc_cat"]),
                        tc_group=str(metadata["tc_group"]),
                        tc_base_s=float(metadata["tc_base_s"]),
                        tc_inc_s=float(metadata["tc_inc_s"]),
                        active_clock=active_clock,
                        opp_clock=opp_clock,
                        active_inc=active_inc,
                        opp_inc=opp_inc,
                        y_val=0.0,
                        policy_move=FLAG_SENTINEL_MOVE,
                        policy_is_resign=False,
                        policy_is_flag=True,
                        time_target=white_last if loser == chess.WHITE else black_last,
                        game_total_plies=int(metadata["total_plies"]),
                    )
                )

    return rows


def write_manifest(config: Config, plan: Plan, rows_written: int) -> None:
    manifest = {
        "input": str(config.input_path),
        "mode": config.mode,
        "selection_mode": config.selection_mode,
        "target_positions": config.target_positions,
        "eligible_rows": plan.eligible_rows,
        "rows_written": rows_written,
        "min_active_elo": config.min_active_elo,
        "max_active_elo": config.max_active_elo,
        "tc_group_filter": config.tc_group,
        "schema_columns": SCHEMA_COLUMNS,
    }
    if config.mode == "test":
        manifest["keep_probability"] = plan.keep_probability
    else:
        manifest["analysis_cell_target"] = plan.cell_target
        manifest["analysis_cell_counts"] = {
            f"elo_bin_{cell[0]}_time_bucket_{cell[1]}": count
            for cell, count in sorted(plan.cell_counts.items())
        }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Stream a Lichess .pgn or .pgn.zst file into parquet shards for test or analysis evaluation datasets."
    )
    parser.add_argument("--input", required=True, help="Path to a .pgn or .pgn.zst file from database.lichess.org")
    parser.add_argument("--output-dir", default=None, help="Output directory for parquet shards and manifest.json")
    parser.add_argument("--mode", choices=["test", "analysis"], required=True, help="test keeps the natural row distribution; analysis balances across Elo and clock-ratio buckets")
    parser.add_argument("--target-positions", type=int, default=DEFAULT_TARGET_POSITIONS, help=f"Approximate target row count (default: {DEFAULT_TARGET_POSITIONS})")
    parser.add_argument("--min-active-elo", type=int, default=ELO_MIN_DEFAULT, help=f"Minimum active-player Elo to keep (default: {ELO_MIN_DEFAULT})")
    parser.add_argument("--max-active-elo", type=int, default=ELO_MAX_DEFAULT, help=f"Exclusive maximum active-player Elo to keep (default: {ELO_MAX_DEFAULT})")
    parser.add_argument("--tc-group", choices=["all", "bullet", "blitz", "rapid", "classical"], default="all", help="Optional game-level time-control filter")
    parser.add_argument("--selection-mode", choices=["stream", "scan"], default="stream", help="stream is one-pass and stops once enough rows are collected; scan is two-pass and samples across the whole file")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Rows per parquet shard (default: {CHUNK_SIZE})")
    parser.add_argument("--seed", type=int, default=20260405, help="Seed for deterministic row sampling")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    default_output_dir = Path("data") / f"{input_path.stem.replace('.pgn', '')}_{args.mode}"
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir

    return Config(
        input_path=input_path,
        output_dir=output_dir,
        mode=args.mode,
        target_positions=args.target_positions,
        min_active_elo=args.min_active_elo,
        max_active_elo=args.max_active_elo,
        tc_group=args.tc_group,
        selection_mode=args.selection_mode,
        seed=args.seed,
        chunk_size=args.chunk_size,
        overwrite=bool(args.overwrite),
    )


def main() -> None:
    config = parse_args()
    ensure_output_dir(config)
    plan = build_plan(config)

    writer = ShardedParquetWriter(config.output_dir, prefix=config.mode, chunk_size=config.chunk_size)
    written_games = 0
    kept_cell_counts: Counter[tuple[int, int]] = Counter()
    progress = tqdm(iter_games(config.input_path), desc="Pass 2/2: write", unit="game")
    for game in progress:
        rows = extract_rows_from_game(game, config, plan, kept_cell_counts)
        if rows:
            writer.add_rows(rows)
            written_games += 1
        current_rows = writer.total_written + len(writer.buffer)
        progress.set_postfix(rows=current_rows, games=written_games)
        if config.selection_mode == "stream" and config.target_positions > 0 and current_rows >= config.target_positions:
            break
    progress.close()

    rows_written = writer.finalize()
    write_manifest(config, plan, rows_written)
    print(f"Wrote {rows_written:,} rows to {config.output_dir}")


if __name__ == "__main__":
    main()
