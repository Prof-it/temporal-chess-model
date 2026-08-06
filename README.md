# Temporal Conditioning for Human Behavioral Cloning in Chess

Code, weights and results for the IEEE GEM 2026 paper.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Behavioral cloning models in chess predict human moves from the board while
ignoring the clock, yet time pressure is a major determinant of how humans play.
This work extends a Chessformer-style encoder with **learned temporal
conditioning tokens** (time control, both players' clocks, recent tempo) plus
behavioral heads for thinking time and resignation.

Temporal conditioning gives a modest overall accuracy gain that **more than
triples under time pressure**, +0.73 pp overall and +2.45 pp in critical time
trouble against an otherwise identical time-blind twin, and it lets the sampled
policy reproduce the human move-quality collapse that a time-blind model
systematically underestimates by around 15 centipawns.

## What the model sees

Alongside the board, the model reads four temporal signals, each encoded as its
own token so attention can use them separately:

| Signal | Concretely |
|---|---|
| Time control | Base time and increment of the game, e.g. 180+2 |
| Your clock | Seconds left on the side to move's clock |
| Opponent's clock | Seconds left on the other clock |
| Recent tempo | How long each of the last eight plies took |

## Install

Playing against the model needs only the ONNX weights and a small runtime
(no PyTorch, about 150 MB installed):

```bash
git clone https://github.com/Prof-it/temporal-chess-model.git
cd temporal-chess-model
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements-engine.txt

python download_assets.py --engine                   # the default model, 89 MB
python download_assets.py --engine --model 79m-cond  # or pick another
```

| Model | Size | |
|---|---|---|
| `23m-cond` | 89 MB | Time-conditioned, 1.7M steps. The default |
| `79m-cond` | 304 MB | Time-conditioned, 1.65M steps. Larger and stronger |
| `23m-blind` | 89 MB | Time-blind, 1M steps. The baseline, for comparison |
| `23m-cond-1m` | 89 MB | Time-conditioned, stopped at 1M steps |
| `79m-cond-1m` | 304 MB | Time-conditioned, stopped at 1M steps |

The `-1m` models are step-matched to `23m-blind`, so those three are the set to
use when comparing conditioned against time-blind. The other two trained longer
and play better.

GPU inference needs `onnxruntime-gpu` and a CUDA / cuDNN 9.x runtime. Without
them the engine runs on CPU by itself.

Training the model or reproducing the paper needs the full stack
(`requirements.txt`) instead: see **[REPRODUCE.md](REPRODUCE.md)**.

## Play against it

The model is a UCI engine, so any chess GUI can use it.

1. Point your GUI at `scripts\marvin-uci.bat` on Windows, or
   `scripts/marvin-uci.sh` on Linux and macOS, as a **UCI engine**. That is the
   whole setup; the launcher finds the repository from its own location.
   - Arena: *Engines, Install New Engine*, pick the file, choose *UCI*
   - Cute Chess: *Tools, Settings, Engines, +*, *Command* is the file, *Protocol* is `UCI`
   - Nibbler: *Engine, Choose engine*, pick the file
2. Set `ActiveElo` and `OpponentElo` in the engine options to choose the rating
   it plays at, from 1200 to 2400.

Play a timed game and the engine picks the clock up from the GUI automatically,
building its temporal context as the game goes. Nothing else to configure.

Whichever models you downloaded are detected automatically, so downloading just
one is enough. With several present the engine prefers `23m-cond`; pick a
different one with `MARVIN_WEIGHTS=inference/marvin_79m_cond_1650k.onnx`.

Sampling, pacing and search options are documented in
[inference/README.md](inference/README.md).

## Repository layout

```
model.py               Maia3Net: backbone + temporal conditioning + behavioral heads
train.py               training loop and CLI
dataset.py             parquet shard loader
download_assets.py     fetch weights and evaluation data

build_dataset/         Lichess to balanced parquet pipeline, and the test-set builder
evaluation/            the harness behind every table and figure
inference/             ONNX runtime, MCTS, UCI engine
scripts/               paper pipeline, engine launchers, ONNX export, release tooling
results/               metrics, tables and figures as reported in the paper
manifests/             dataset and checkpoint provenance
tests/                 smoke tests and the checkpoint regression gate
```

## Citation

```bibtex
@inproceedings{dziwniel2026temporal,
  author    = {Victor Dziwniel and Tianxiang Lu},
  title     = {Temporal Conditioning for Human Behavioral Cloning in Chess},
  booktitle = {2026 IEEE Gaming, Entertainment and Media Conference (GEM)},
  year      = {2026},
  publisher = {IEEE}
}
```

## Acknowledgments

- [Lichess](https://database.lichess.org) for the open game database
- The [CSSLab Maia-3](https://github.com/CSSLab/maia3) team, whose Chessformer
  backbone this reproduces and extends, and whose released checkpoints are the
  comparison baseline
- [Stockfish](https://stockfishchess.org) and
  [python-chess](https://python-chess.readthedocs.io)

## License

AGPL-3.0
