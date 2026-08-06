"""Evaluate the official released Maia-3 checkpoints on our parquet eval sets.

Loads CSSLab's released weights with their own model code (vendored at
``tools/maia3_official``) and converts our parquet rows into their input
format, so the comparison runs on identical, row-aligned positions.

Format correspondence (verified against tools/maia3_official/maia3/dataset.py):
  - Both codebases canonicalize each ply to *that ply's* side-to-move via
    ``chess.Board.mirror()`` and share the piece code map (P..K = 1..6, +6 for
    the opponent). Our ``board_history`` is newest-first with trailing all-zero
    planes when the game is short; Maia-3 concatenates oldest-first and pads by
    repeating the earliest real position (their inference behaviour), which we
    reproduce here.
  - Their policy is 4352 logits: 64*64 from-to scores plus 256 promotion
    slots ordered ``4096 + (from_file*8 + to_file)*4 + piece`` with piece
    q,r,b,n = 0..3 (our promo codes 1..4 minus one). Verified against
    ``get_all_possible_moves()`` at startup.
  - Move-matching follows evaluation/metrics.py: resign/flag rows are skipped,
    each legal move is scored, and a prediction is correct iff its (from, to)
    squares match the human move. NLL is the softmax over scored legal moves
    (exact triplet, promotion piece included).

``--allie-filters`` reproduces the Chessformer paper's test filtering on our
unfiltered Allie set: drop the first 10 moves (ply < 20) and every position
from the first time either player is below 30 seconds (game-level cut). Used
to reproduce their reported accuracy before trusting the model on our set.

Usage:
    python -m evaluation.eval_maia3_official --model maia3-79m \
        --eval-set data/testset_2026-05 \
        --out eval_results/paper_2026/rows_maia3_79m_testset2026.parquet
    python -m evaluation.eval_maia3_official --model maia3-79m \
        --eval-set data/allie_testset/allie_test_set.parquet --allie-filters \
        --out eval_results/paper_2026/rows_maia3_79m_allie_filtered.parquet
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAIA3_DIR = _REPO_ROOT / "tools" / "maia3_official"
sys.path.insert(0, str(_MAIA3_DIR))

try:
    from maia3.models import MAIA3Model                      # noqa: E402
    from maia3.model_registry import MODEL_SPECS, resolve_checkpoint_path  # noqa: E402
    from maia3.utils import get_all_possible_moves           # noqa: E402
except ModuleNotFoundError as exc:                           # pragma: no cover
    # Maia-3 is deliberately not vendored: the baseline should be their code and
    # their weights, fetched from the source at a pinned commit.
    raise SystemExit(
        f"{exc}\n\n"
        f"The released Maia-3 baseline is not installed. Clone it (see REPRODUCE.md):\n"
        f"    git clone https://github.com/CSSLab/maia3 {_MAIA3_DIR}\n"
        f"    cd {_MAIA3_DIR} && git checkout 1e13597\n"
    ) from exc

from .data import resolve_base_time, resolve_eval_files, bucket_index  # noqa: E402

HISTORY = 8
N_FROMTO = 64 * 64
N_MOVES = 4352

_COLUMNS = [
    "game_id", "ply", "board_history", "legal_moves", "active_elo", "opp_elo",
    "active_clock", "opp_clock", "active_inc", "opp_inc", "tc_base_s", "tc_inc_s",
    "tc_cat", "base_time", "y_val", "policy_move", "policy_is_resign", "policy_is_flag",
]


def _verify_move_index_layout() -> None:
    """Assert their flat move list matches the logits layout we index into."""
    import chess
    all_moves = get_all_possible_moves()
    assert len(all_moves) == N_MOVES, len(all_moves)
    for f, t in ((0, 0), (12, 28), (63, 7)):
        expect = chess.square_name(f) + chess.square_name(t)
        assert all_moves[f * 64 + t] == expect, (f, t, all_moves[f * 64 + t])
    pieces = "qrbn"
    for ff in (0, 3, 7):
        for tf in (0, 4):
            for p in range(4):
                idx = 4096 + (ff * 8 + tf) * 4 + p
                expect = (chess.square_name(chess.square(ff, 6))
                          + chess.square_name(chess.square(tf, 7)) + pieces[p])
                assert all_moves[idx] == expect, (idx, all_moves[idx], expect)


def maia3_move_index(from_sq: int, to_sq: int, promo: int) -> int:
    """Map our canonical (from, to, promo-code) triplet to their logit index."""
    if promo == 0:
        return from_sq * 64 + to_sq
    return 4096 + ((from_sq - 48) * 8 + (to_sq - 56)) * 4 + (promo - 1)


def rows_to_tokens(rows: list[dict]) -> np.ndarray:
    """Convert our newest-first piece-code planes to Maia-3 (B, 64, 96) tokens."""
    out = np.empty((len(rows), 64, 12 * HISTORY), dtype=np.float32)
    channels = np.arange(1, 13, dtype=np.int64)
    for i, row in enumerate(rows):
        planes = np.asarray(row["board_history"], dtype=np.int64)  # (8, 64) newest-first
        real = planes.any(axis=1)
        n_real = int(real.sum())
        # Guard: real planes are contiguous from index 0 (newest); all-zero
        # planes only appear as trailing padding.
        if n_real == 0:
            raise ValueError("row with empty board history")
        # Oldest-first order with their repeat-earliest padding.
        idx_seq = [n_real - 1] * (HISTORY - n_real) + list(range(n_real - 1, -1, -1))
        onehot = (planes[idx_seq][:, :, None] == channels).astype(np.float32)  # (8, 64, 12)
        out[i] = onehot.transpose(1, 0, 2).reshape(64, 12 * HISTORY)
    return out


def load_official_model(alias: str, device: str, checkpoint_path: str | None = None):
    import torch
    spec = next((s for s in MODEL_SPECS if s.name == alias or alias in s.aliases), None)
    if spec is None:
        raise SystemExit(f"Unknown Maia-3 alias {alias!r}; "
                         f"use one of {[s.name for s in MODEL_SPECS]}")
    cfg = types.SimpleNamespace(**spec.config, device=device)
    model = MAIA3Model(cfg)
    path = checkpoint_path or resolve_checkpoint_path(spec)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = {k.replace("smolgen", "gab"): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"State dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[maia3] loaded {spec.display_name} ({n_params/1e6:.1f}M params) from {path}")
    return model


def apply_allie_filters(rows: list[dict], min_ply: int = 10) -> list[dict]:
    """Chessformer test filters on one game's rows (rows sorted by ply):
    drop the first 10 plies, and everything from the first sub-30s clock on.

    ``min_ply=10`` reproduces their curated Allie count to within 0.5%
    (879,532 vs their 884,049 of our 1,239,353 unfiltered rows), so "first 10
    moves" in the paper reads as 10 plies.
    """
    kept = []
    for row in sorted(rows, key=lambda r: int(r["ply"])):
        if float(row["active_clock"]) < 30.0 or float(row["opp_clock"]) < 30.0:
            break
        if int(row["ply"]) < min_ply:
            continue
        kept.append(row)
    return kept


def iter_row_groups(files, allie_filters: bool, limit: int | None, min_ply: int = 10):
    """Yield rows, optionally applying the game-level Allie filter.

    Games are assumed contiguous within shards (asserted); the per-game buffer
    is flushed whenever game_id changes.
    """
    seen_games: set = set()
    emitted = 0
    buffer: list[dict] = []
    current_game = None

    def _flush():
        nonlocal buffer
        rows = apply_allie_filters(buffer, min_ply) if allie_filters else buffer
        buffer = []
        return rows

    for path in files:
        pf = pq.ParquetFile(str(path))
        avail = set(pf.schema_arrow.names)
        cols = [c for c in _COLUMNS if c in avail]
        for rb in pf.iter_batches(batch_size=20_000, columns=cols):
            for row in rb.to_pylist():
                gid = row["game_id"]
                if gid != current_game:
                    if buffer:
                        for r in _flush():
                            yield r
                            emitted += 1
                            if limit is not None and emitted >= limit:
                                return
                    assert gid not in seen_games, f"game {gid} not contiguous"
                    seen_games.add(gid)
                    current_game = gid
                buffer.append(row)
    for r in _flush():
        yield r
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="maia3-79m")
    p.add_argument("--checkpoint-path", default=None,
                   help="Local .pt override (otherwise downloaded/cached from HF)")
    p.add_argument("--eval-set", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allie-filters", action="store_true",
                   help="Apply the Chessformer paper's test filters "
                        "(drop first 10 plies and post-sub-30s positions)")
    p.add_argument("--min-ply", type=int, default=10,
                   help="Ply cutoff used with --allie-filters")
    args = p.parse_args()

    import torch

    _verify_move_index_layout()
    files, set_label = resolve_eval_files(args.eval_set)
    model = load_official_model(args.model, args.device, args.checkpoint_path)
    use_amp = args.device.startswith("cuda")

    out_rows: dict[str, list] = {k: [] for k in (
        "game_id", "ply", "active_elo", "bucket_index", "clock_ratio",
        "active_clock", "opp_clock",
        "tc_base_s", "tc_inc_s", "tc_cat", "correct", "correct_exact", "nll",
        "wdl_correct", "n_legal", "target_from", "target_to", "pred_from", "pred_to",
    )}

    n_seen = n_skipped = 0
    batch_rows: list[dict] = []

    def _process(batch: list[dict]) -> None:
        tokens = torch.from_numpy(rows_to_tokens(batch)).to(args.device)
        self_elos = torch.tensor([float(r["active_elo"]) for r in batch], device=args.device)
        oppo_elos = torch.tensor([float(r["opp_elo"]) for r in batch], device=args.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            logits_move, logits_value, _ = model(tokens, self_elos, oppo_elos)
        logits_move = logits_move.float().cpu().numpy()
        wdl_pred = logits_value.float().argmax(dim=1).cpu().numpy()

        for i, row in enumerate(batch):
            legal = row["legal_moves"]
            idxs = np.fromiter(
                (maia3_move_index(int(m[0]), int(m[1]), int(m[2])) for m in legal),
                dtype=np.int64, count=len(legal))
            scores = logits_move[i, idxs]
            best = int(np.argmax(scores))
            pf_, pt_, pp_ = (int(v) for v in legal[best])
            tf_, tt_, tp_ = (int(v) for v in row["policy_move"])
            # NLL over the scored legal moves, exact triplet as target.
            logz = np.logaddexp.reduce(scores - scores.max()) + scores.max()
            target_pos = next((j for j, m in enumerate(legal)
                               if int(m[0]) == tf_ and int(m[1]) == tt_ and int(m[2]) == tp_), None)
            nll = float(logz - scores[target_pos]) if target_pos is not None else float("nan")

            base_time = resolve_base_time(row)
            ratio = min(max(float(row["active_clock"]) / max(base_time, 1.0), 0.0), 1.0)
            y_val = int(round(float(row["y_val"]) * 2))

            out_rows["game_id"].append(row["game_id"])
            out_rows["ply"].append(int(row["ply"]))
            out_rows["active_elo"].append(int(row["active_elo"]))
            out_rows["bucket_index"].append(bucket_index(ratio))
            out_rows["clock_ratio"].append(ratio)
            out_rows["tc_base_s"].append(float(row.get("tc_base_s") or base_time))
            out_rows["tc_inc_s"].append(float(row.get("tc_inc_s") or row.get("active_inc") or 0.0))
            out_rows["tc_cat"].append(int(row.get("tc_cat", -1)))
            out_rows["correct"].append(bool(pf_ == tf_ and pt_ == tt_))
            out_rows["correct_exact"].append(bool(pf_ == tf_ and pt_ == tt_ and pp_ == tp_))
            out_rows["active_clock"].append(float(row["active_clock"]))
            out_rows["opp_clock"].append(float(row["opp_clock"]))
            out_rows["nll"].append(nll)
            out_rows["wdl_correct"].append(bool(int(wdl_pred[i]) == y_val))
            out_rows["n_legal"].append(len(legal))
            out_rows["target_from"].append(tf_)
            out_rows["target_to"].append(tt_)
            out_rows["pred_from"].append(pf_)
            out_rows["pred_to"].append(pt_)

    from tqdm import tqdm
    progress = tqdm(desc=f"maia3 {set_label}", unit="pos")
    for row in iter_row_groups(files, args.allie_filters, args.limit, args.min_ply):
        if row.get("policy_is_resign") or row.get("policy_is_flag"):
            n_skipped += 1
            continue
        batch_rows.append(row)
        n_seen += 1
        progress.update(1)
        if len(batch_rows) >= args.batch_size:
            _process(batch_rows)
            batch_rows = []
    if batch_rows:
        _process(batch_rows)
    progress.close()

    table = pa.table(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)

    correct = np.asarray(out_rows["correct"])
    buckets = np.asarray(out_rows["bucket_index"])
    print(f"\n[maia3] {set_label}: n={len(correct):,} (skipped resign/flag={n_skipped:,})")
    print(f"[maia3] top-1 accuracy overall: {correct.mean()*100:.2f}%")
    for b, name in enumerate(("Comfortable", "Moderate", "Pressured", "Critical")):
        m = buckets == b
        if m.any():
            print(f"[maia3]   {name:11s}: {correct[m].mean()*100:.2f}%  (n={int(m.sum()):,})")
    print(f"[maia3] wrote {out_path}")


if __name__ == "__main__":
    main()
