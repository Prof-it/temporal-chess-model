# Reproducing the paper

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
apt install stockfish                        # move-quality stages only

python download_assets.py --checkpoints --test-set

# Maia-3 baseline: their code and their weights, at a pinned commit
git clone https://github.com/CSSLab/maia3 tools/maia3_official
git -C tools/maia3_official checkout 1e13597
```

## Run

```bash
scripts/run_paper_pipeline.sh              # everything, in order
scripts/run_paper_pipeline.sh --list       # stage names
scripts/run_paper_pipeline.sh --only bands # one stage
```

Environment overrides: `PY`, `CKPT_DIR`, `EVAL_SET`, `OUT`, `STOCKFISH`,
`ENGINE_WORKERS`, `ALLIE_SET`.

**Run stages one at a time.** Two concurrent CUDA jobs exhaust a 16 GB card.

Curated outputs land in `$OUT/{metrics,tables,figures}`; the heavy per-row
parquets stay at the top of `$OUT`.

## Stages

| Stage | Paper element | Cost |
|---|---|---|
| `tables` | Table I (top block), Table II | ~25 min |
| `timehead` | Table III, think-time move-conditionality | ~35 min |
| `rows` | per-position outputs for all 7 model/transform arms | ~3 h |
| `bands` | Table I (rating-band block) | ~2 min, CPU |
| `comparison` | Figure 2, system-comparison table | ~5 min, CPU |
| `ladder` | Table IV | ~5 min, CPU |
| `maia3repro` | Maia-3's published 57.1%, as a harness check | ~1 h |
| `seedsweep` | Table I seed-sweep row | ~1.5 h, needs training first |
| `quality` | Figure 1a source (Stockfish) | hours |
| `counterfactual` | Figure 1b source + dose-response (Stockfish) | hours |
| `figures` | `quality.png`, `comparison.png` | seconds |

Timings are for one RTX 5080 with 12 Stockfish workers. `bands`, `comparison`
and `ladder` are CPU-only, but they read the per-position parquets that `rows`
writes, so run that stage first.

Each stage is a thin wrapper over one `python -m evaluation.*` call. Read
`scripts/run_paper_pipeline.sh` for the exact arguments.

## Training

Both arms of the ablation, identical but for one flag:

```bash
# time-conditioned, 1,000,000 steps
python train.py --config 23m --batch-size 256 --grad-accum-steps 4 --lr 1.25e-4 \
    --seed 67 --data-seed 42 --data-dir data/train --val-data-dir data/val

# time-blind twin: add --no-time-conditioning
# 79M: --config 79m --batch-size 128 --grad-accum-steps 8 --lr 1e-4
```

Seed sweep: six runs stopped at 250k steps, seeds 67/68/69 with data seeds
42/43/44. `--lr-total-steps` keeps them on the 1M-step LR trajectory, so they are
truncations of the headline run rather than differently-scheduled runs:

```bash
python train.py --config 23m --batch-size 256 --grad-accum-steps 4 --lr 1.25e-4 \
    --seed 68 --data-seed 43 --max-steps 250000 --lr-total-steps 1758335 \
    --data-dir data/train --val-data-dir data/val [--no-time-conditioning]
```

Pair `s67` is the 250k archive of the headline runs, so it needs no separate run.
Checkpoints archive every 250k steps.

The six seed-sweep checkpoints are not distributed, only the five in
`download_assets.py`. Running the `seedsweep` stage means training them with the
commands above first.

Single RTX 5080 (16 GB), fp16 mixed precision, AdamW (weight decay 0.01,
gradient clipping 1.0), cosine decay after 2,000 linear warmup steps, effective
batch size 1024. Loss weights: policy 1.0, WDL 0.5, thinking time 0.5, promotion
0.1, resignation 0.1, time-forfeit 0.1; positive-class weights 20 (resignation)
and 50 (time-forfeit).

## Data

**Training set**, 1.8B positions. Streams Lichess monthly dumps over HTTP
without staging PGN to disk, extracts, balances across Elo bins and time
controls, shuffles, verifies. Resumable.

```bash
python -m build_dataset --status
python -m build_dataset --end 2025-12
python -m build_dataset --verify-only
```

**Test set**, 1,500,029 positions from May 2026:

```bash
python -m build_dataset.build_testset --mode test \
    --input data/raw_2026/2026-05_prefix.pgn.zst \
    --output-dir data/testset_2026-05 --target-positions 1500000 \
    --min-active-elo 1200 --max-active-elo 2500 --seed 20260405
```

Prefer `download_assets.py --test-set` for the exact shards: a rebuild draws
from the same dump with the same seed, but bit-exactness across a re-download is
not guaranteed, and every published number is computed on the shipped shards.

## Verifying an install

```bash
python -m tests.smoke_model      # parameter counts, forward pass, time-blindness
python -m tests.smoke_encoding   # parquet-row encoding, clock reaches the policy
```

With checkpoints downloaded, the regression gate scores each one on a slice and
fails on any drift. Run it before and after touching `model.py`, `train.py` or
`dataset.py`:

```bash
python -m tests.gate_checkpoints --limit 20000 --out /tmp/gate.json
python -m tests.gate_checkpoints --limit 20000 --baseline /tmp/gate.json
```
