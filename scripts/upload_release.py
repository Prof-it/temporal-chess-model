"""Upload the staged checkpoints and evaluation data to Hugging Face.

Creates the repo if it does not exist and uploads two folders:

    weights/      14 checkpoints + 3 ONNX exports        (~6.1 GB)
    eval-data/    the May-2026 test set + per-row outputs  (~330 MB)

plus ``checkpoints_MANIFEST.json`` and a model card at the repo root.

Authenticate first (``hf auth login`` or ``export HF_TOKEN=...``), then:

    python -m scripts.upload_release --staging-dir staging --private

The repo id must match ``REPO_ID`` in download_assets.py. Uploads are
resumable: rerun after an interruption and only missing files transfer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_CARD = """---
license: agpl-3.0
tags:
  - chess
  - behavioral-cloning
  - human-ai-alignment
---

# Temporal Conditioning for Human Behavioral Cloning in Chess

Weights and evaluation data for the IEEE GEM 2026 paper.

Code and reproduction instructions:
https://github.com/Prof-it/temporal-chess-model

## Models

Five checkpoints, each released as `.onnx` (to play) and `.pt` (to evaluate).

| Model | Steps | |
|---|---|---|
| `marvin_23m_cond_1700k` | 1.70M | 23M time-conditioned, longest-trained and strongest |
| `marvin_79m_cond_1650k` | 1.65M | 79M time-conditioned, the model the paper compares to Maia-3 |
| `marvin_23m_cond_1M` | 1.00M | 23M time-conditioned, ablation arm |
| `marvin_23m_blind_1M` | 1.00M | 23M time-blind twin, ablation baseline |
| `marvin_79m_cond_1M` | 1.00M | 79M time-conditioned, step-matched to the 23M pair |

The three `1M` checkpoints are step-matched, so those are the ones to compare
against each other. Training of the time-blind arm stopped at 1M steps, which is
why it has no later checkpoint.

`checkpoints_MANIFEST.json` records the role, sha256, step count, seeds,
parameter count and training command for each.

## Evaluation data

`eval-data/testset_2026-05/` is the held-out test set every reported number is
computed on: 1,500,029 positions from May 2026 Lichess games, one position per
game, rated 1200-2500, across all time controls.

## Use

```bash
git clone https://github.com/Prof-it/temporal-chess-model
cd temporal-chess-model
pip install -r requirements-engine.txt
python download_assets.py --engine
scripts/marvin-uci.sh          # or point a chess GUI at it
```

Every `.pt` embeds its own `config_dict`, so `evaluation.backends.make_backend`
loads any of them without being told the architecture.

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
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo-id", default="holymolyyy/temporal-chess-model",
                   help="Must match REPO_ID in download_assets.py")
    p.add_argument("--staging-dir", default="staging",
                   help="Directory holding weights/ and eval-data/")
    p.add_argument("--manifest", default="manifests/checkpoints_MANIFEST.json")
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--public", dest="private", action="store_false",
                   help="Create/leave the repo public")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would upload and exit")
    args = p.parse_args()

    from huggingface_hub import HfApi

    staging = Path(args.staging_dir)
    weights, eval_data = staging / "weights", staging / "eval-data"
    for d in (weights, eval_data):
        if not d.is_dir():
            raise SystemExit(f"missing staging directory: {d}  "
                             f"(run scripts/stage_release.py first)")

    files = sorted(weights.rglob("*")) + sorted(eval_data.rglob("*"))
    files = [f for f in files if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} files, {total / 2**30:.2f} GiB -> {args.repo_id} "
          f"({'private' if args.private else 'public'})")
    for f in files:
        print(f"  {f.relative_to(staging)}  ({f.stat().st_size / 2**20:.0f} MiB)")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as exc:
        raise SystemExit(
            f"Not authenticated with Hugging Face ({type(exc).__name__}).\n"
            "Run `hf auth login`, or export HF_TOKEN=<a write token from "
            "https://huggingface.co/settings/tokens>.") from exc
    print(f"\nauthenticated as {who.get('name')}")

    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    print(f"repo ready: https://huggingface.co/{args.repo_id}")

    # Model card and manifest first: cheap, and they make a partially-uploaded
    # repo self-describing if a large transfer is interrupted.
    api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")
    manifest = Path(args.manifest)
    if manifest.exists():
        json.loads(manifest.read_text())          # fail early on a corrupt manifest
        api.upload_file(path_or_fileobj=str(manifest),
                        path_in_repo="checkpoints_MANIFEST.json",
                        repo_id=args.repo_id, repo_type="model")
    print("uploaded README.md + checkpoints_MANIFEST.json")

    for folder, prefix in ((weights, "weights"), (eval_data, "eval-data")):
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        print(f"\nuploading {prefix}/ ({size / 2**30:.2f} GiB) …")
        api.upload_folder(folder_path=str(folder), path_in_repo=prefix,
                          repo_id=args.repo_id, repo_type="model")
        print(f"  {prefix}/ done")

    print(f"\nhttps://huggingface.co/{args.repo_id}")
    if args.private:
        print("Repo is PRIVATE. Make it public from the repo settings when ready.")


if __name__ == "__main__":
    sys.exit(main())
