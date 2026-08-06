"""Counterfactual clock injection: isolate the *causal* effect of the clock.

Observational blunder-rate-by-bucket confounds the clock with position type
(time-pressure buckets skew toward sharp endgames), so a board-conditioned
time-blind model rides the same ramp without ever seeing the clock. This probe
removes the confound: it holds each position fixed and re-runs the model with
the clock inputs overridden to a range of remaining-time fractions. Any change
in sampled move quality is then attributable to the clock signal alone.

For each position and each injected clock fraction f, the active and opponent
clocks are set to f * base_time (all other inputs unchanged), a move is sampled,
and its CPL is scored with Stockfish. We report blunder rate / mean CPL vs the
injected fraction, on a fixed position set — plus the human's actual move as a
reference. The time-conditioned model's curve should rise as f drops; the
time-blind model's should be flat.

Usage:
    python -m evaluation.counterfactual --models cond=<pt> agn=<pt> \
        --eval-set data/testset_2026-05 --fracs 1.0,0.5,0.2,0.05,0.02 \
        --limit 20000 --depth 12 --engine-workers 14 --out eval_results/paper_2026/cf.parquet
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
from .data import resolve_base_time
from .engine import (
    reconstruct_actual_board, canonical_move_to_actual_move,
    score_legal_moves, compute_cpls,
)

_KEYS = ("board_history", "time_history", "rep_flags", "castling",
         "ep_mask", "scalars", "legal_mask")


def _encode(row, base_time, clock_override=None):
    """Encode a row; if clock_override (seconds) given, set both clocks to it."""
    active = float(row["active_clock"]) if clock_override is None else float(clock_override)
    opp = float(row["opp_clock"]) if clock_override is None else float(clock_override)
    b = make_model_batch_from_parquet_row(
        board_history=row["board_history"], legal_moves=row["legal_moves"],
        repetition_flags=row["repetition_flags"], time_history_s=row["time_history"],
        castling_rights=row["castling_rights"], ep_square=row["ep_square"],
        active_elo=row["active_elo"], opponent_elo=row["opp_elo"],
        move_ply=row["move_ply"], halfmove_clock=row["halfmove_clock"],
        active_clock_s=active, opponent_clock_s=opp,
        active_inc_s=row["active_inc"], opponent_inc_s=row["opp_inc"],
        base_time_s=base_time)
    return {k: np.asarray(b[k])[0] for k in _KEYS}


def _sample_uci(outputs, i, row, rng, temperature, top_p):
    move_logits, _wdl, _rs, _fl, _tb, _tf, _tt, promo_logits = outputs
    scored = score_legal_moves(move_logits[i], promo_logits[i], row["legal_moves"])
    if not scored:
        return None
    res = sample_from_logits(np.asarray([s[1] for s in scored], dtype=np.float64),
                             temperature=temperature, top_p=top_p, rng=rng)
    s = scored[res.move_index]
    return canonical_move_to_actual_move(s[2], s[3], s[4], int(row["move_ply"])).uci()


def run(files, model_specs, out_path, *, fracs, device="cuda", depth=12, limit=20000,
        temperature=1.0, top_p=0.98, engine_workers=14, cpl_cap=1500, mate_cp=2000,
        engine_path="stockfish", seed=0, batch_size=256, keep_buckets=None):
    backends = {lab: make_backend(p, device=device, config_name=lab) for lab, p in model_specs}
    rngs = {lab: np.random.default_rng(seed + k * 1009) for k, lab in enumerate(backends)}

    from tqdm import tqdm
    pbar = tqdm(total=limit, desc="Model pass", unit="pos")
    tasks, metas = [], []
    buf = []

    def flush():
        if not buf:
            return
        for lab, be in backends.items():
            rng = rngs[lab]
            # One forward pass per injected fraction (+ real clock).
            variants = {"real": None}
            variants.update({f"f{f:g}": f for f in fracs})
            sampled = {name: [] for name in variants}
            for name, f in variants.items():
                enc = [_encode(r, r["_base"], None if f is None else f * r["_base"]) for r in buf]
                batch = {k: np.stack([e[k] for e in enc]) for k in _KEYS}
                outs = be(batch, return_promo=True)
                for i, r in enumerate(buf):
                    sampled[name].append(_sample_uci(outs, i, r, rng, temperature, top_p))
            for i, r in enumerate(buf):
                if lab == list(backends)[0]:
                    # emit one task-set per position (shared board); human included once
                    human = canonical_move_to_actual_move(
                        int(r["policy_move"][0]), int(r["policy_move"][1]),
                        int(r["policy_move"][2]), int(r["move_ply"])).uci()
                    r["_moves"] = {"human": human}
                    r["_board"] = reconstruct_actual_board(SimpleNamespace(
                        board_history=r["board_history"], move_ply=int(r["move_ply"]),
                        castling_rights=r["castling_rights"], ep_square=int(r["ep_square"]),
                        halfmove_clock=int(r["halfmove_clock"]))).fen()
                for name in variants:
                    r["_moves"][f"{lab}_{name}"] = sampled[name][i]
        for r in buf:
            tasks.append((r["_board"], r["_moves"]))
            metas.append({"bucket": r["_bucket"], "real_frac": r["_real_frac"]})
            pbar.update(1)
        buf.clear()

    stop = False
    for path in files:
        if stop:
            break
        pf = pq.ParquetFile(str(path))
        avail = set(pf.schema_arrow.names)
        cols = [c for c in ["board_history", "legal_moves", "time_history", "active_elo",
                "opp_elo", "move_ply", "halfmove_clock", "repetition_flags", "castling_rights",
                "ep_square", "base_time", "tc_base_s", "active_clock", "opp_clock", "active_inc",
                "opp_inc", "policy_move", "policy_is_resign", "policy_is_flag"] if c in avail]
        for rb in pf.iter_batches(batch_size=20000, columns=cols):
            for row in rb.to_pylist():
                if limit and len(tasks) + len(buf) >= limit:
                    stop = True
                    break
                if row.get("policy_is_resign") or row.get("policy_is_flag"):
                    continue
                base = resolve_base_time(row)
                rf = min(max(float(row["active_clock"]) / max(base, 1.0), 0.0), 1.0)
                bk = (0 if rf > 0.5 else 1 if rf > 0.2 else 2 if rf > 0.05 else 3)
                if keep_buckets is not None and bk not in keep_buckets:
                    continue
                row["_base"] = base
                row["_real_frac"] = rf
                row["_bucket"] = bk
                buf.append(row)
                if len(buf) >= batch_size:
                    flush()
            if stop:
                break
    flush()
    pbar.close()

    move_names = list(tasks[0][1].keys())
    print(f"[counterfactual] {len(tasks)} positions x {len(move_names)} moves -> Stockfish")
    cpls = compute_cpls(tasks, engine_path=engine_path, engine_depth=depth, mate_cp=mate_cp,
                        cpl_cap=cpl_cap, engine_workers=engine_workers, engine_hash_mb=256,
                        engine_threads_per_worker=1)
    cols = {"bucket": [m["bucket"] for m in metas], "real_frac": [m["real_frac"] for m in metas]}
    for name in move_names:
        cols[f"cpl_{name}"] = [r.get(name) for r in cpls]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: pa.array(v) for k, v in cols.items()}), str(out_path))
    print(f"[counterfactual] wrote {out_path}")

    # Quick summary: blunder rate vs injected fraction, per model.
    def rate(name, thr=300):
        v = np.array([np.nan if x is None else float(x) for x in cols[f"cpl_{name}"]])
        v = v[~np.isnan(v)]
        return 100 * (np.maximum(v, 0) > thr).mean()
    print("\nBlunder rate (%) vs injected clock (fixed positions):")
    print("  human (real move):", round(rate("human"), 1))
    for lab in backends:
        row = "  ".join(f"{n}={rate(f'{lab}_{n}'):.1f}" for n in
                        ["real"] + [f"f{f:g}" for f in fracs])
        print(f"  {lab}: {row}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Counterfactual clock-injection probe")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--eval-set", required=True)
    p.add_argument("--fracs", default="1.0,0.5,0.2,0.05,0.02")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--engine-workers", type=int, default=14)
    p.add_argument("--engine-path", default="stockfish",
                   help="Stockfish binary (name on PATH or absolute path)")
    p.add_argument("--buckets", default=None,
                   help="Comma-separated real-bucket indices to keep (0=Comf..3=Crit)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from .data import resolve_eval_files
    files, _ = resolve_eval_files(args.eval_set)
    specs = [s.split("=", 1) for s in args.models]
    specs = [(a.strip(), b.strip()) for a, b in specs]
    fracs = [float(x) for x in args.fracs.split(",")]
    keep_buckets = ({int(x) for x in args.buckets.split(",")} if args.buckets else None)
    run(files, specs, Path(args.out), fracs=fracs, device=args.device, depth=args.depth,
        limit=args.limit, temperature=args.temperature, top_p=args.top_p,
        engine_workers=args.engine_workers, engine_path=args.engine_path,
        seed=args.seed, keep_buckets=keep_buckets)


if __name__ == "__main__":
    main()
