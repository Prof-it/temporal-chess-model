"""Stage the paper's checkpoints for upload and write their manifest.

Hard-links each checkpoint under a release name (so staging costs no extra
disk), records sha256 / step count / seeds / parameter counts / the training
command that produced it, and writes ``checkpoints_MANIFEST.json``.

Uploads nothing. Point a Hugging Face upload at the staging directory
afterwards, then fill in ``REPO_ID`` in download_assets.py.

    python -m scripts.stage_release --checkpoints-dir checkpoints \
        --staging-dir staging/weights --manifest manifests/checkpoints_MANIFEST.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from model import Maia3Net, count_parameters

# Release name -> (source path relative to --checkpoints-dir, role, seed, data_seed)
RELEASES = [
    # Paper checkpoints: the step-matched 1M pair the controlled ablation uses.
    ("marvin_23m_cond_1M.pt",
     "archive/chessformer_maia3-23m_elo-endpoints_time-tokens_tempotok_s67_step1000000.pt",
     "23M time-conditioned, 1M steps. Headline arm of the controlled ablation "
     "(Tables I-III, Fig. 1).", 67, 42),
    ("marvin_23m_blind_1M.pt",
     "archive/chessformer_23m_elo-endpoints_time-blind_s67_step1000000.pt",
     "23M time-blind twin, 1M steps. Identical backbone, data and schedule; "
     "receives zeroed temporal inputs. Training stopped here, so this is also "
     "the latest time-blind checkpoint.", 67, 42),
    ("marvin_79m_cond_1M.pt",
     "archive/chessformer_79m_elo-endpoints_time-tokens_tempotok_s67_step1000000.pt",
     "79M time-conditioned at 1M steps, for step-matched comparison with the "
     "23M pair.", 67, 42),

    # Longest-trained checkpoints. Stronger than the 1M pair, but past the
    # step-matched budget, so no reported number uses them.
    ("marvin_23m_cond_1700k.pt",
     "chessformer_maia3-23m_elo-endpoints_time-tokens_tempotok_s67_latest.pt",
     "23M time-conditioned, 1.7M steps. Longest-trained 23M and the strongest "
     "by validation loss.", 67, 42),
    ("marvin_79m_cond_1650k.pt",
     "chessformer_79m_elo-endpoints_time-tokens_tempotok_s67_latest.pt",
     "79M time-conditioned, 1.65M steps. The 79M model the paper evaluates "
     "against released Maia-3 (Fig. 2, Table IV).", 67, 42),
]


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def training_command(config: dict, step: int, seed: int, data_seed: int) -> str:
    """Reconstruct the train.py invocation for a checkpoint."""
    size = "79m" if config.get("dim_vit") == 1024 else "23m"
    lr = "1e-4" if size == "79m" else "1.25e-4"
    parts = [f"python train.py --config {size}",
             "--batch-size 256 --grad-accum-steps 4",   # effective batch 1024
             f"--lr {lr}", f"--seed {seed}", f"--data-seed {data_seed}",
             "--data-dir data/train --val-data-dir data/val"]
    if config.get("no_time_conditioning"):
        parts.append("--no-time-conditioning")
    if step == 250_000:
        # Truncated seed-sweep runs stay on the 1M-step LR trajectory.
        parts.append("--max-steps 250000 --lr-total-steps 1758335")
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoints-dir", default="checkpoints")
    p.add_argument("--staging-dir", default="staging/weights")
    p.add_argument("--manifest", default="manifests/checkpoints_MANIFEST.json")
    p.add_argument("--skip-hash", action="store_true",
                   help="Skip sha256 (fast dry run; the manifest then omits it)")
    args = p.parse_args()

    root = Path(args.checkpoints_dir)
    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    entries = []
    for release, rel, role, seed, data_seed in RELEASES:
        src = root / rel
        if not src.exists():
            print(f"  MISSING  {release}  <- {rel}")
            continue

        dst = staging / release
        if dst.exists():
            dst.unlink()
        try:
            dst.hardlink_to(src)
            link = "hardlink"
        except OSError:                       # different filesystem
            dst.symlink_to(src.resolve())
            link = "symlink"

        ck = torch.load(src, map_location="cpu", weights_only=False)
        config = ck.get("config_dict", {})
        step = int(ck.get("global_step", 0))
        params = count_parameters(Maia3Net(dict(config)))

        entry = {
            "file": release,
            "role": role,
            "source_filename": src.name,
            "size_bytes": src.stat().st_size,
            "steps": step,
            "seed": seed,
            "data_seed": data_seed,
            "parameters": params,
            "time_conditioned": not config.get("no_time_conditioning", False),
            "config_dict": config,
            "train_command": training_command(config, step, seed, data_seed),
        }
        if not args.skip_hash:
            entry["sha256"] = sha256(src)
        entries.append(entry)
        print(f"  {link:8s} {release:30s} {step:>9,} steps  {params:>12,} params")

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "description": "Checkpoints for 'Temporal Conditioning for Human "
                       "Behavioral Cloning in Chess' (IEEE GEM 2026).",
        "note": "Parameter counts are unique parameters. A naive sum over "
                "state_dict() is 4,194,304 larger because the GAB weight is "
                "shared across all 8 blocks and counted 8 times.",
        "checkpoints": entries,
    }, indent=2))
    print(f"\nStaged {len(entries)} checkpoints in {staging}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
