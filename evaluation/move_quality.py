"""Centipawn-loss (CPL) move-quality export for the paper's Figs 1-2.

For a balanced sample of positions this:
  1. samples a move from each model's policy at a given temperature,
  2. scores the position, the human move, and each model's sampled move with
     Stockfish, yielding signed CPL (>=0, capped) from the mover's perspective,
  3. buckets by time pressure and categorizes CPL (Perfect..Blunder).

Output is a tidy parquet consumed by :mod:`evaluation.figures`:
    bucket_index, rating_band, human_cpl, <model>_cpl (one col per model),
    plus the sampled move ucis for auditing.

The centerpiece claim — the time-conditioned policy's CPL tracks the human
degradation under pressure while the time-agnostic policy stays too strong —
is read directly off this table.

NOTE: CPL requires a Stockfish-annotatable position set; for a publishable
number that set must be *date-disjoint* from training (see the 2026 test set).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from inference.encoding import make_model_batch_from_parquet_row
from inference.sampling import sample_from_logits
from .backends import make_backend
from .data import bucket_index, rating_band_index, resolve_base_time
from .engine import (
    reconstruct_actual_board, canonical_move_to_actual_move,
    score_legal_moves, compute_cpls,
)

# CPL -> quality category (cp thresholds; matches scripts/plot_analysis_topk_cpl.py).
QUALITY_CATEGORIES = ("Perfect", "Good", "Inaccuracy", "Mistake", "Blunder")


def cpl_category(cpl: float) -> str:
    if cpl <= 0:
        return "Perfect"
    if cpl <= 25:
        return "Good"
    if cpl <= 100:
        return "Inaccuracy"
    if cpl <= 300:
        return "Mistake"
    return "Blunder"


def _sampled_move(outputs, i, row, rng, temperature, top_p):
    """Argmax-scored legal moves -> sample one; return an actual chess.Move."""
    move_logits, _wdl, _rs, _fl, _tb, _tf, _tt, promo_logits = outputs
    scored = score_legal_moves(move_logits[i], promo_logits[i], row["legal_moves"])
    if not scored:
        return None
    res = sample_from_logits(
        np.asarray([s[1] for s in scored], dtype=np.float64),
        temperature=temperature, top_p=top_p, rng=rng)
    s = scored[res.move_index]
    return canonical_move_to_actual_move(s[2], s[3], s[4], int(row["move_ply"]))


def _keys():
    return ("board_history", "time_history", "rep_flags", "castling",
            "ep_mask", "scalars", "legal_mask")


def _collect_tasks(
    files: list[Path], backends: dict, rngs: dict, *,
    limit: int | None, per_cell_cap: int | None,
    temperature: float, top_p: float, batch_size: int,
    keep_buckets: set[int] | None = None,
    keep_tc: set[int] | None = None,
):
    """GPU pass: run models over a balanced position sample, returning Stockfish
    tasks ``(fen, {name: uci})`` plus row-aligned metadata."""
    tasks: list[tuple] = []
    metas: list[dict] = []
    cell_counts: dict[tuple[int, int], int] = {}
    labels = list(backends)

    from tqdm import tqdm
    pbar = tqdm(total=limit, desc="Model pass", unit="pos")
    buffer: list[dict] = []

    def _flush():
        if not buffer:
            return
        batch = {}
        enc = []
        for row in buffer:
            b = make_model_batch_from_parquet_row(
                board_history=row["board_history"], legal_moves=row["legal_moves"],
                repetition_flags=row["repetition_flags"], time_history_s=row["time_history"],
                castling_rights=row["castling_rights"], ep_square=row["ep_square"],
                active_elo=row["active_elo"], opponent_elo=row["opp_elo"],
                move_ply=row["move_ply"], halfmove_clock=row["halfmove_clock"],
                active_clock_s=row["active_clock"], opponent_clock_s=row["opp_clock"],
                active_inc_s=row["active_inc"], opponent_inc_s=row["opp_inc"],
                base_time_s=resolve_base_time(row))
            enc.append({k: np.asarray(b[k])[0] for k in _keys()})
        batch = {k: np.stack([e[k] for e in enc]) for k in _keys()}
        outs = {label: be(batch, return_promo=True) for label, be in backends.items()}
        for i, row in enumerate(buffer):
            board = reconstruct_actual_board(SimpleNamespace(
                board_history=row["board_history"], move_ply=int(row["move_ply"]),
                castling_rights=row["castling_rights"], ep_square=int(row["ep_square"]),
                halfmove_clock=int(row["halfmove_clock"])))
            human = canonical_move_to_actual_move(
                int(row["policy_move"][0]), int(row["policy_move"][1]),
                int(row["policy_move"][2]), int(row["move_ply"]))
            move_map = {"human": human.uci()}
            sampled_uci = {}
            for label in labels:
                mv = _sampled_move(outs[label], i, row, rngs[label], temperature, top_p)
                move_map[label] = mv.uci() if mv else None
                sampled_uci[label] = move_map[label]
            tasks.append((board.fen(), move_map))
            metas.append({"bucket_index": row["_bucket"], "rating_band": row["_band"],
                          "active_elo": int(row["active_elo"]), "human_uci": human.uci(),
                          **{f"{l}_uci": sampled_uci[l] for l in labels}})
            pbar.update(1)
        buffer.clear()

    stop = False
    for parquet_path in files:
        if stop:
            break
        pf = pq.ParquetFile(str(parquet_path))
        avail = set(pf.schema_arrow.names)
        want = ["board_history", "legal_moves", "time_history", "active_elo", "opp_elo",
                "move_ply", "halfmove_clock", "repetition_flags", "castling_rights",
                "ep_square", "base_time", "tc_base_s", "tc_cat", "active_clock", "opp_clock",
                "active_inc", "opp_inc", "policy_move", "policy_is_resign", "policy_is_flag"]
        use = [c for c in want if c in avail]
        for rb in pf.iter_batches(batch_size=20000, columns=use):
            for row in rb.to_pylist():
                if limit is not None and len(tasks) + len(buffer) >= limit:
                    stop = True
                    break
                if row.get("policy_is_resign") or row.get("policy_is_flag"):
                    continue
                base = resolve_base_time(row)
                ratio = min(max(float(row["active_clock"]) / max(base, 1.0), 0.0), 1.0)
                b = bucket_index(ratio)
                # Cheap pre-model filter: skip whole buckets before encoding, so a
                # large pressured/critical sample doesn't require a full model pass
                # over the (dominant) comfortable bucket.
                if keep_buckets is not None and b not in keep_buckets:
                    continue
                tc = int(row.get("tc_cat", -1))
                if keep_tc is not None and tc not in keep_tc:
                    continue
                band = rating_band_index(int(row["active_elo"]))
                if per_cell_cap is not None:
                    # Balance over elo x bucket, and additionally over TC when a
                    # TC filter is active (so bullet can't dominate a cell).
                    key = (band, b, tc) if keep_tc is not None else (band, b)
                    if cell_counts.get(key, 0) >= per_cell_cap:
                        continue
                    cell_counts[key] = cell_counts.get(key, 0) + 1
                row["_bucket"] = b
                row["_band"] = band
                buffer.append(row)
                if len(buffer) >= batch_size:
                    _flush()
            if stop:
                break
    _flush()
    pbar.close()
    return tasks, metas, labels


def export_cpl(
    files: list[Path],
    model_specs: list[tuple[str, str]],
    out_path: Path,
    *,
    device: str = "cuda",
    depth: int = 12,
    limit: int | None = 20000,
    per_cell_cap: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    cpl_cap: int = 1500,
    mate_cp: int = 2000,
    engine_path: str = "stockfish",
    engine_workers: int = 8,
    engine_hash_mb: int = 256,
    seed: int = 0,
    batch_size: int = 256,
    keep_buckets: set[int] | None = None,
    keep_tc: set[int] | None = None,
) -> Path:
    """Two-phase CPL export: GPU model pass builds Stockfish tasks, then a
    parallel worker pool scores them (reusing the tested ``compute_cpls``)."""
    backends = {label: make_backend(path, device=device, config_name=label)
                for label, path in model_specs}
    rngs = {label: np.random.default_rng(seed + (k + 1) * 1009)
            for k, label in enumerate(backends)}

    tasks, metas, labels = _collect_tasks(
        files, backends, rngs, limit=limit, per_cell_cap=per_cell_cap,
        temperature=temperature, top_p=top_p, batch_size=batch_size,
        keep_buckets=keep_buckets, keep_tc=keep_tc)
    print(f"[move_quality] {len(tasks)} positions -> Stockfish "
          f"(depth={depth}, workers={engine_workers})")

    cpl_results = compute_cpls(
        tasks, engine_path=engine_path, engine_depth=depth, mate_cp=mate_cp,
        cpl_cap=cpl_cap, engine_workers=engine_workers,
        engine_hash_mb=engine_hash_mb, engine_threads_per_worker=1)

    cols = {"bucket_index": [], "rating_band": [], "active_elo": [],
            "human_cpl": [], "human_uci": []}
    for label in labels:
        cols[f"{label}_cpl"] = []
        cols[f"{label}_uci"] = []
    for meta, res in zip(metas, cpl_results):
        cols["bucket_index"].append(meta["bucket_index"])
        cols["rating_band"].append(meta["rating_band"])
        cols["active_elo"].append(meta["active_elo"])
        cols["human_uci"].append(meta["human_uci"])
        cols["human_cpl"].append(res.get("human"))
        for label in labels:
            cols[f"{label}_uci"].append(meta[f"{label}_uci"])
            cols[f"{label}_cpl"].append(res.get(label))

    table = pa.table({k: pa.array(v) for k, v in cols.items()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    print(f"[move_quality] wrote {len(cols['human_cpl'])} rows -> {out_path}")
    return out_path


def _parse_specs(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        label, path = s.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export CPL move-quality table (Figs 1-2)")
    p.add_argument("--models", nargs="+", required=True, help="label=path specs")
    p.add_argument("--eval-set", required=True, help="dir of parquet shards")
    p.add_argument("--out", required=True, help="output parquet path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--per-cell-cap", type=int, default=None,
                   help="balance: max rows per (rating_band x bucket) cell")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--engine-path", default="stockfish")
    p.add_argument("--engine-workers", type=int, default=8,
                   help="Parallel Stockfish worker threads (main throughput lever)")
    p.add_argument("--buckets", default=None,
                   help="Comma-separated bucket indices to keep (0=Comf,1=Mod,2=Press,"
                        "3=Crit); pre-model filter to oversample pressure cheaply")
    p.add_argument("--tc", default=None,
                   help="Comma-separated tc_cat to keep (0=bullet,1=blitz,2=rapid,"
                        "3=classical); also balances the per-cell-cap across TC")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    keep_buckets = ({int(x) for x in args.buckets.split(",")} if args.buckets else None)
    keep_tc = ({int(x) for x in args.tc.split(",")} if args.tc else None)

    from .data import resolve_eval_files
    files, label = resolve_eval_files(args.eval_set)
    print(f"[move_quality] set={label} shards={len(files)} depth={args.depth} "
          f"temp={args.temperature} keep_buckets={keep_buckets}")
    export_cpl(
        files, _parse_specs(args.models), Path(args.out),
        device=args.device, depth=args.depth, limit=args.limit,
        per_cell_cap=args.per_cell_cap, temperature=args.temperature,
        top_p=args.top_p, engine_path=args.engine_path,
        engine_workers=args.engine_workers, seed=args.seed,
        keep_buckets=keep_buckets, keep_tc=keep_tc)


if __name__ == "__main__":
    main()
