"""Download model weights and the evaluation test set from Hugging Face.

    python download_assets.py --engine                  # ONNX weights, ready to play
    python download_assets.py --engine --model 23m-cond # just one of them
    python download_assets.py --checkpoints             # PyTorch checkpoints
    python download_assets.py --test-set                # the May-2026 evaluation set

ONNX weights land in ``inference/``, checkpoints in ``checkpoints/``, and the
test set in ``data/testset_2026-05/``.

While the Hugging Face repo is private, authenticate first with ``hf auth login``
or by setting ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "holymolyyy/temporal-chess-model"

# Five models, each released as both .onnx (to play) and .pt (to evaluate).
#
# The 1M checkpoints are the step-matched set the paper's controlled ablation
# uses. The 1700k and 1650k checkpoints continued past that budget: stronger,
# but not what any reported number was computed on. Training of the time-blind
# arm stopped at 1M, so it has no later checkpoint.
MODELS = {
    "23m-cond": "marvin_23m_cond_1700k",
    "23m-cond-1m": "marvin_23m_cond_1M",
    "23m-blind": "marvin_23m_blind_1M",
    "79m-cond": "marvin_79m_cond_1650k",
    "79m-cond-1m": "marvin_79m_cond_1M",
}

DEFAULT_ENGINE_MODEL = "23m-cond"

TEST_SET = {
    "eval-data/testset_2026-05/manifest.json": "data/testset_2026-05",
    **{f"eval-data/testset_2026-05/test_{i:05d}.parquet": "data/testset_2026-05"
       for i in range(7)},
}


def fetch(files: dict[str, str], root: Path) -> None:
    """Download ``{remote path: local directory}`` into ``root``."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError,
                                        RepositoryNotFoundError)

    for name, dest in files.items():
        out_dir = root / dest
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / Path(name).name
        if target.exists():
            print(f"  have  {target}")
            continue
        print(f"  get   {name} -> {target}")
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=name)
        except (GatedRepoError, RepositoryNotFoundError) as exc:
            raise SystemExit(
                f"\nCould not read {REPO_ID}: {type(exc).__name__}.\n"
                "If the repo is still private, authenticate first:\n"
                "    hf auth login          (or export HF_TOKEN=...)\n") from exc
        except EntryNotFoundError as exc:
            raise SystemExit(
                f"\n{name} is not in {REPO_ID}.\n"
                "Check the model name, or see the repo for what it currently "
                "holds:\n"
                f"    https://huggingface.co/{REPO_ID}/tree/main\n") from exc
        target.write_bytes(Path(path).read_bytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", action="store_true",
                   help="ONNX weights for the UCI engine")
    p.add_argument("--checkpoints", action="store_true",
                   help="PyTorch checkpoints, for evaluation and fine-tuning")
    p.add_argument("--model", action="append", choices=sorted(MODELS),
                   help="Limit --engine / --checkpoints to these models "
                        f"(repeatable). Default for --engine: {DEFAULT_ENGINE_MODEL}")
    p.add_argument("--test-set", action="store_true",
                   help="The 1.5M-position May-2026 evaluation set (~80 MB)")
    p.add_argument("--all", action="store_true", help="Every model, plus the test set")
    p.add_argument("--root", default=".", help="Repo root to download into")
    args = p.parse_args()

    if not any((args.engine, args.checkpoints, args.test_set, args.all)):
        p.print_help()
        sys.exit(1)

    root = Path(args.root).resolve()

    if args.engine or args.all:
        wanted = args.model or (sorted(MODELS) if args.all else [DEFAULT_ENGINE_MODEL])
        print("ONNX weights:")
        fetch({f"weights/{MODELS[m]}.onnx": "inference" for m in wanted}, root)
    if args.checkpoints or args.all:
        wanted = args.model or sorted(MODELS)
        print("Checkpoints:")
        fetch({**{f"weights/{MODELS[m]}.pt": "checkpoints" for m in wanted},
               "checkpoints_MANIFEST.json": "checkpoints"}, root)
    if args.test_set or args.all:
        print("Test set:")
        fetch(TEST_SET, root)
    print("\nDone.")


if __name__ == "__main__":
    main()
