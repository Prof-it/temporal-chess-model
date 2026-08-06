"""Decompose the counterfactual injection into clock vs tempo contributions.

On every pressured+critical position, sample cond moves under three temporal
inputs and score blunder rate with Stockfish, plus human and time-blind
references at the real clock:

  cpl_human               human's actual move
  cpl_cond_real           cond at the real clock and tempo (matches human)
  cpl_agn_real            time-blind at the real board (no clock)
  cpl_cond_comfclk        cond, clock injected to full base time, tempo kept real
  cpl_cond_comfclk_tempo  cond, clock AND tempo set to a comfortable regime

The comfortable tempo is donated from a random comfortable-bucket position of
the same time control (its recorded 8-ply move times), so the manipulation uses
a real, in-distribution tempo pattern. The staircase cond_real > cond_comfclk >
cond_comfclk_tempo isolates the marginal contribution of each temporal signal.
"""
from __future__ import annotations
import argparse
import glob
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.backends import make_backend
from evaluation.data import resolve_base_time
from evaluation.engine import (
    reconstruct_actual_board, canonical_move_to_actual_move, score_legal_moves, compute_cpls)
from inference.sampling import sample_from_logits
from inference.encoding import make_model_batch_from_parquet_row

_KEYS = ("board_history","time_history","rep_flags","castling","ep_mask","scalars","legal_mask")


def _encode(row, base_time, clk=None, tempo=None):
    active = float(row["active_clock"]) if clk is None else float(clk)
    opp = float(row["opp_clock"]) if clk is None else float(clk)
    th = row["time_history"] if tempo is None else tempo
    b = make_model_batch_from_parquet_row(
        board_history=row["board_history"], legal_moves=row["legal_moves"],
        repetition_flags=row["repetition_flags"], time_history_s=th,
        castling_rights=row["castling_rights"], ep_square=row["ep_square"],
        active_elo=row["active_elo"], opponent_elo=row["opp_elo"],
        move_ply=row["move_ply"], halfmove_clock=row["halfmove_clock"],
        active_clock_s=active, opponent_clock_s=opp,
        active_inc_s=row["active_inc"], opponent_inc_s=row["opp_inc"], base_time_s=base_time)
    return {k: np.asarray(b[k])[0] for k in _KEYS}


def _sample(outs, i, row, rng):
    ml, _w, _r, _fl, _tb, _tf, _tt, pr = outs
    scored = score_legal_moves(ml[i], pr[i], row["legal_moves"])
    if not scored:
        return None
    res = sample_from_logits(np.asarray([s[1] for s in scored], dtype=np.float64),
                             temperature=1.0, top_p=0.98, rng=rng)
    s = scored[res.move_index]
    return canonical_move_to_actual_move(s[2], s[3], s[4], int(row["move_ply"])).uci()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", nargs="+", required=True,
                   help="Exactly two label=path specs: cond=<ckpt> agn=<ckpt>")
    p.add_argument("--eval-set", default="data/testset_2026-05")
    p.add_argument("--out", default="eval_results/paper_2026/counterfactual_tempo.parquet")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--engine-workers", type=int, default=12)
    p.add_argument("--engine-path", default="stockfish",
                   help="Stockfish binary (name on PATH or absolute path)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    specs = dict(s.split("=", 1) for s in args.models if "=" in s)
    if set(specs) != {"cond", "agn"}:
        raise SystemExit("--models must be exactly: cond=<ckpt> agn=<ckpt>")

    cols = ["board_history","legal_moves","time_history","active_elo","opp_elo","move_ply",
            "halfmove_clock","repetition_flags","castling_rights","ep_square","tc_base_s",
            "tc_group","active_clock","opp_clock","active_inc","opp_inc","policy_move",
            "policy_is_resign","policy_is_flag"]
    donors, pc = {}, []
    for f in sorted(glob.glob(f"{args.eval_set}/*.parquet")):
        for rb in pq.ParquetFile(f).iter_batches(batch_size=20000, columns=cols):
            for row in rb.to_pylist():
                if row["policy_is_resign"] or row["policy_is_flag"]:
                    continue
                base = resolve_base_time(row)
                rf = float(row["active_clock"]) / max(base, 1.0)
                if rf > 0.5:
                    donors.setdefault(row["tc_group"], []).append(list(row["time_history"]))
                elif rf <= 0.2 and (args.limit is None or len(pc) < args.limit):
                    row["_base"] = base
                    row["_bucket"] = 2 if rf > 0.05 else 3
                    pc.append(row)
    print(f"[cf-tempo] pressured+critical n={len(pc)}  donors={{{', '.join(f'{k}:{len(v)}' for k,v in donors.items())}}}")

    cond = make_backend(specs["cond"], device=args.device, config_name="cond")
    agn = make_backend(specs["agn"], device=args.device, config_name="agn")
    rng = np.random.default_rng(args.seed)
    # pre-draw a comfortable-tempo donor per position (fixed, reproducible)
    for r in pc:
        pool = donors.get(r["tc_group"]) or next(iter(donors.values()))
        r["_comftempo"] = pool[rng.integers(len(pool))]

    arms = {  # name -> (backend, clock_frac, use_comftempo)
        "cond_real": (cond, None, False),
        "agn_real": (agn, None, False),
        "cond_comfclk": (cond, 1.0, False),
        "cond_comfclk_tempo": (cond, 1.0, True),
    }
    moves = {name: [None]*len(pc) for name in arms}
    from tqdm import tqdm
    for name, (be, frac, ct) in arms.items():
        r_ = np.random.default_rng(args.seed + hash(name) % 9973)
        for s in tqdm(range(0, len(pc), args.batch_size), desc=name, unit="batch"):
            chunk = pc[s:s+args.batch_size]
            enc = []
            for r in chunk:
                clk = None if frac is None else frac * r["_base"]
                tempo = r["_comftempo"] if ct else None
                enc.append(_encode(r, r["_base"], clk, tempo))
            batch = {k: np.stack([e[k] for e in enc]) for k in _KEYS}
            o = be(batch, return_promo=True)
            for i, r in enumerate(chunk):
                moves[name][s+i] = _sample(o, i, r, r_)

    tasks, bucket = [], []
    for j, r in enumerate(pc):
        board = reconstruct_actual_board(SimpleNamespace(
            board_history=r["board_history"], move_ply=int(r["move_ply"]),
            castling_rights=r["castling_rights"], ep_square=int(r["ep_square"]),
            halfmove_clock=int(r["halfmove_clock"]))).fen()
        human = canonical_move_to_actual_move(int(r["policy_move"][0]), int(r["policy_move"][1]),
                                              int(r["policy_move"][2]), int(r["move_ply"])).uci()
        mv = {"human": human}
        for name in arms:
            mv[name] = moves[name][j]
        tasks.append((board, mv)); bucket.append(r["_bucket"])
    names = ["human", *arms.keys()]
    print(f"[cf-tempo] scoring {len(tasks)} positions x {len(names)} moves (depth {args.depth})...")
    cpls = compute_cpls(tasks, engine_path=args.engine_path, engine_depth=args.depth, mate_cp=2000,
                        cpl_cap=1500, engine_workers=args.engine_workers, engine_hash_mb=256,
                        engine_threads_per_worker=1)
    out = {"bucket": bucket}
    for nm in names:
        out[f"cpl_{nm}"] = [c.get(nm) for c in cpls]
    pq.write_table(pa.table({k: pa.array(v) for k, v in out.items()}), args.out)
    print(f"[cf-tempo] wrote {args.out}")

    meta = np.array(bucket)
    def rate(nm, mask):
        v = np.array([np.nan if c.get(nm) is None else float(c[nm]) for c in cpls])
        valid = mask & ~np.isnan(v)
        return 100*(np.maximum(v[valid], 0) > 300).mean(), int(valid.sum())
    print("\nBlunder rate (%):")
    for lbl, m in [("Pressured", meta==2), ("Critical", meta==3), ("Both", meta>=2)]:
        print(f"  {lbl}:")
        for nm in names:
            r_, n = rate(nm, m)
            print(f"    {nm:22s} {r_:5.2f}  [n={n}]")
        b_, _ = rate("cond_comfclk", m); c_, _ = rate("cond_comfclk_tempo", m); a_, _ = rate("cond_real", m)
        print(f"    clock effect (real->comfclk): {a_-b_:+.2f}   tempo effect (comfclk->+tempo): {b_-c_:+.2f}")


if __name__ == "__main__":
    main()
