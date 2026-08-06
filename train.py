"""Trainer for the Maia3 chess model using per-token input encoding and legal move masking."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter

from dataset import create_dataloader, count_samples

# Import model (Maia3 is the canonical architecture).
from model import (
    Maia3Net,
    CONFIG_MAIA3_3M,
    CONFIG_MAIA3_5M,
    CONFIG_MAIA3_23M,
    CONFIG_MAIA3_79M,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maia3 trainer")
    parser.add_argument("--data-dir", default="data/train",
                        help="Root containing train/ subdir of parquet shards. "
                             "Defaults to the balanced training set with base_time.")
    parser.add_argument("--config", choices=["3m", "5m", "23m", "79m"], default="5m")
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many optimizer steps (across epochs). Useful for "
                             "bounding diagnostic runs to a fixed budget. Default: no limit.")
    parser.add_argument("--lr", type=float, default=1.25e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=50000, help="Run validation every N optimizer steps")
    parser.add_argument("--val-batches", type=int, default=20000, help="Max batches per validation run (0 for full)") # 5000 for batch size 1024, so 20_000 for batch size 256*4 grad accum?
    parser.add_argument("--checkpoint-interval", type=int, default=10000, help="Save checkpoint every N optimizer steps (for resume)")
    parser.add_argument("--checkpoint-archive-interval", type=int, default=250000, help="Save non-overwriting checkpoints every N optimizer steps")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-archive-dir", default="checkpoints/archive")
    parser.add_argument("--prefetch-factor", type=int, default=4)  # Lower but more workers
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="max-autotune",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: max-autotune)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory usage")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--cyclic-lr", action="store_true",
                        help="Use cyclic cosine annealed LR schedule. "
                             "--lr sets max LR, --min-lr sets floor. Cycle length = total_steps / --cyclic-num-cycles.")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Minimum learning rate for cyclic schedule (default: 1e-5)")
    parser.add_argument("--cyclic-num-cycles", type=int, default=50000,
                        help="Number of cosine cycles over the full training run (default: 50000)")
    parser.add_argument("--lr-total-steps", type=int, default=None,
                        help="Override the total_steps used to shape the cosine LR decay curve "
                             "(normally args.epochs * ceil(train_samples/batch_size)). By default "
                             "this is recomputed from the CURRENT dataset size every run, so if "
                             "--data-dir's contents change between launches, a --resume'd run's decay "
                             "curve silently reshapes instead of continuing the original trajectory. "
                             "Set this explicitly when resuming a checkpoint to lock the schedule to "
                             "the exact total_steps the checkpoint was trained under (read off the "
                             "original run's TensorBoard train/lr curve or recomputed offline) — "
                             "critical for seed-sweep / ablation runs that must stay on the same LR "
                             "trajectory as the run they forked from.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-wait", type=int, default=2)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-active", type=int, default=6)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=67,
                        help="Training seed: model init + torch/numpy/random RNG state at process "
                             "start (set_seed). Does NOT affect data loading order — see --data-seed.")
    parser.add_argument("--data-seed", type=int, default=42,
                        help="Data-loader seed: controls StreamingDataset file/row shuffle order and "
                             "per-worker RNG seeding (dataset.py default was previously hardcoded here, "
                             "decoupled from --seed). Kept separate from --seed so ablation arms can be "
                             "compared on identical batch order (same --data-seed, different --seed) or, "
                             "for a full seed sweep, varied independently to capture data-order variance too.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable bit-level determinism (cuDNN deterministic, no benchmark, "
                             "torch.use_deterministic_algorithms, CUBLAS workspace pinned). "
                             "Required for clean within-family ablation comparisons. "
                             "Expect ~5-15%% throughput hit.")
    parser.add_argument("--time-encoder",
                        choices=["tokens", "mlp", "channels", "interp_channels", "film", "bias"],
                        default="tokens",
                        help="Time conditioning variant. 'tokens' = log-binned tokens "
                             "prepended (baseline). 'mlp' = continuous MLP broadcast-added "
                             "to all 64 squares. 'channels' (maia3 only) = per-signal "
                             "log-binned embeddings concatenated as input channels. "
                             "'interp_channels' (maia3 only) = Elo-style scalar endpoint "
                             "interpolation concatenated as input channels (continuous; the "
                             "closest mirror of how elo is conditioned; width set by "
                             "--time-interp-channel-dim). "
                             "'film' (maia3 only) = per-sample (gamma, beta) modulation. "
                             "'bias' (maia3 only) = time-conditioned (n_heads, 64, 64) "
                             "attention bias added to GAB each block.")
    parser.add_argument("--time-interp-channel-dim", type=int, default=None,
                        help="Per-signal channel width for --time-encoder interp_channels. "
                             "Default (None) = dim_vit, so each time signal gets the SAME "
                             "width it would as a prepended token — a clean tokens-vs-channels "
                             "comparison (identical info + scaling + per-signal capacity). "
                             "Set a smaller value only to deliberately shrink the time footprint; "
                             "too small swamps the signal (the earlier failure mode). Ignored "
                             "for other --time-encoder variants.")
    parser.add_argument("--time-featurize", choices=["log1p", "sqrt", "linear", "fourier"],
                        default="log1p",
                        help="Sub-axis for --time-encoder mlp: how raw seconds enter the MLP. "
                             "'log1p' (default) compresses; 'sqrt' less aggressive; 'linear' null "
                             "hypothesis; 'fourier' multi-scale sin/cos at 4 log-spaced periods. "
                             "Ignored when --time-encoder=tokens.")
    parser.add_argument("--per-position-clocks", action=argparse.BooleanOptionalAction, default=False,
                        help="Reconstruct per-history-position clocks for both players from "
                             "time_history + current clocks + increments, log-compress, and "
                             "broadcast into the per-square state input. OFF by default (the default "
                             "now routes time history through the tempo token; see --tempo-history-token). "
                             "Pass --per-position-clocks to re-enable.")
    parser.add_argument("--keep-global-time-encoder", action=argparse.BooleanOptionalAction, default=True,
                        help="Only meaningful with --per-position-clocks. When True (default), the "
                             "global time encoder keeps all 6 inputs (base_time, increment, my_clock, "
                             "opp_clock, my_last_time, opp_last_time) on top of the per-position clocks. "
                             "Pass --no-keep-global-time-encoder to shrink the global encoder to "
                             "{base_time, increment} only.")
    parser.add_argument("--per-position-time-deltas", action=argparse.BooleanOptionalAction, default=False,
                        help="Only meaningful with --per-position-clocks. When True, the per-position "
                             "channels carry per-move clock DELTAS (move durations / tempo) instead of "
                             "absolute remaining clocks — exposes opponent tempo directly. Off by "
                             "default (baseline = absolute clocks).")
    parser.add_argument("--tempo-history-token", action=argparse.BooleanOptionalAction, default=True,
                        help="maia3 interp tokens only. Route the full 8-ply time_history through one "
                             "tempo token (replaces the my_last/opp_last per-move tokens). ON by default "
                             "(default config = tempo token + --no-per-position-clocks). Pass "
                             "--no-tempo-history-token for the legacy my_last/opp_last tokens.")
    parser.add_argument("--zero-time-history", action=argparse.BooleanOptionalAction, default=False,
                        help="Ablation (maia3): feed zeros for time_history, removing ALL per-move "
                             "tempo/history input (my_last/opp_last tokens and per-position channels go "
                             "constant) while keeping current clocks + TC + the time head. Tells you "
                             "whether the model uses tempo at all. Off by default.")
    parser.add_argument("--no-time-conditioning", action="store_true", default=False,
                        help="Time-blind ablation (maia3): build the FULL model — time encoder, time "
                             "head, and flag head all present and identically shaped — but feed constant "
                             "(zero) time signals (base_time, increment, clocks, time_history) so the "
                             "model cannot condition on time. The ONLY difference vs the time-conditioned "
                             "model is the information reaching it, not the parameters/shape, so the time "
                             "head still trains and its loss is directly comparable. Distinct from "
                             "no_time_context, which deletes the time + flag heads entirely. Off by default.")
    # --- GAB Q/K rotation ablation ---------------------------------------
    # Arm A (additive-only, baseline):  --gab-additive-bias    --no-gab-qk-rotation  (defaults)
    # Arm B (additive + rotation):      --gab-additive-bias    --gab-qk-rotation
    # Arm C (rotation-only):            --no-gab-additive-bias --gab-qk-rotation
    parser.add_argument("--gab-qk-rotation", action=argparse.BooleanOptionalAction, default=False,
                        help="Have GAB emit per-square, per-head 2D-plane rotation angles and rotate "
                             "q/k before attention (a dynamic, board-conditional generalization of "
                             "RoPE): logit[i,j] = (R_i q_i)·(R_j k_j). Adds a bilinear content×geometry "
                             "term the additive bias cannot express. Zero-init -> identity at step 0. "
                             "Off by default.")
    parser.add_argument("--gab-additive-bias", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep the additive GAB bias B[i,j] (the standard Smolgen output). ON by "
                             "default. Pass --no-gab-additive-bias together with --gab-qk-rotation for "
                             "the rotation-only arm (Arm C).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Loss weights
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--wdl-weight", type=float, default=0.5)
    parser.add_argument("--time-weight", type=float, default=0.5)
    parser.add_argument("--promo-weight", type=float, default=0.1)
    parser.add_argument("--resign-weight", type=float, default=0.1,
                        help="Loss weight for the resign binary head.")
    parser.add_argument("--flag-weight", type=float, default=0.1,
                        help="Loss weight for the flag (time-forfeit) binary head.")
    parser.add_argument("--resign-pos-weight", type=float, default=20.0,
                        help="BCE positive-class weight for the resign head (counters class imbalance).")
    parser.add_argument("--flag-pos-weight", type=float, default=50.0,
                        help="BCE positive-class weight for the flag head (counters class imbalance).")
    parser.add_argument("--history-mask-prob", type=float, default=0.05,
                        help="Probability a training sample has a uniformly-random number of its "
                             "oldest board-history positions masked (keeps no-history states in-distribution).")
    # Fine-tuning options
    parser.add_argument("--finetune", action="store_true",
                        help="Enable fine-tuning mode: loads only model weights (no optimizer/scheduler state), "
                             "uses discriminative learning rates for backbone vs heads")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint to load model weights from (finetune mode only). "
                             "If not set, uses --resume path.")
    parser.add_argument("--lr-backbone", type=float, default=5e-6,
                        help="Learning rate for backbone (embeddings + transformer layers) in finetune mode")
    parser.add_argument("--lr-heads", type=float, default=5e-5,
                        help="Learning rate for output heads (policy, value, time, etc.) in finetune mode")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Number of transformer layers to freeze (from the bottom). 0 = freeze none.")
    parser.add_argument("--dropout-override", type=float, default=None,
                        help="Override dropout rate in model config (e.g. 0.2 for fine-tuning)")
    parser.add_argument("--mix-general-data", type=str, default=None,
                        help="Path to general training data dir. Mixes general data each epoch "
                             "to prevent catastrophic forgetting (ratio set by --mix-ratio).")
    parser.add_argument("--mix-ratio", type=float, default=0.2,
                        help="Fraction of each epoch that comes from general data (0.0-1.0). Default: 0.2")
    parser.add_argument("--num-val-files", type=int, default=10,
                        help="Number of files to use for validation (default: 10, ~2.5M samples)")
    parser.add_argument("--train-tc", type=str, default=None,
                        choices=["bullet", "blitz", "rapid", "classical", "ultraBullet"],
                        help="Diagnostic: train only on samples matching this time-control category. "
                             "Drops throughput proportionally to the rejection rate (e.g., blitz ≈ 20%% of "
                             "the new dataset → 5× fewer samples/sec, but per-step compute unchanged). "
                             "Useful for isolating 'is this gap dataset-shift or model-quality'.")
    parser.add_argument("--val-tc", type=str, default=None,
                        choices=["bullet", "blitz", "rapid", "classical", "ultraBullet"],
                        help="Like --train-tc but for the val loader. If you want a clean blitz-vs-blitz "
                             "comparison, set both --train-tc blitz and --val-tc blitz.")
    parser.add_argument("--val-data-dir", type=str, default="data/val",
                        help="Separate directory for validation data (must contain train/ subdir with parquet files). "
                             "When set, ALL parquet files in val-data-dir/train/ are used as validation "
                             "(no hash split — no shards silently dropped). "
                             "Pass --val-data-dir '' to fall back to hash-splitting --data-dir.")
    # Per-ELO loss weighting
    parser.add_argument("--elo-loss-weights", type=str, default=None,
                        help="13 comma-separated floats: per-bin loss weights for ELO bins 1200-1300 through 2400-2500. "
                             "Normalized to mean=1.0. Example: '1.5,1.4,1.3,1.2,1.1,1.0,1.0,0.9,0.8,0.7,0.7,0.6,0.6'")
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Pin cuDNN to deterministic kernels and disable autotuner-driven kernel
        # selection (otherwise two runs with identical seeds can still diverge
        # by step ~100 because cuDNN picks different algorithms).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # `warn_only=True` lets ops without a deterministic kernel fall back
        # rather than crashing the run; the warning lands in the log.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required for deterministic CUBLAS reductions on Ampere+ when the
        # split-K threshold is hit. Setting via env var because cuBLAS reads
        # it at handle creation, which happens lazily.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        print(f"[seed] Deterministic mode enabled (seed={seed}) — expect 5-15% throughput hit.")


def configure_precision(device: torch.device, disable_tf32: bool, deterministic: bool = False) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = not disable_tf32
        torch.backends.cudnn.allow_tf32 = not disable_tf32
        # Skip cudnn.benchmark when --deterministic is set; set_seed has already
        # forced benchmark=False + deterministic=True, and flipping benchmark
        # back to True here would silently undo that.
        if not deterministic:
            torch.backends.cudnn.benchmark = True


def create_profiler(args, run_dir: Path, device: torch.device, available_steps: Optional[int] = None):
    if not args.profile:
        return nullcontext()

    wait = args.profile_wait
    warmup = args.profile_warmup
    active = args.profile_active

    if available_steps is not None:
        cycle = wait + warmup + active
        if cycle > available_steps:
            wait = min(wait, max(0, available_steps - 2))
            warmup = min(warmup, max(0, available_steps - wait - 1))
            active = max(1, available_steps - wait - warmup)
            print(f"[profiler] Adjusted to wait={wait}, warmup={warmup}, active={active}")

    trace_dir = Path(args.profile_dir) if args.profile_dir else run_dir / "profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"[profiler] Traces: {trace_dir}")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    return profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=args.profile_repeat),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=args.profile_stack,
    )


def move_batch_to_device(batch: Dict, device: torch.device, non_blocking: bool = False) -> Dict:
    """Move all tensors in batch to device."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, dict):
            result[k] = {kk: vv.to(device, non_blocking=non_blocking) for kk, vv in v.items()}
        else:
            result[k] = v
    return result


def assemble_time_logits(time_baseline, time_from_logits, time_to_logits, policy_target):
    """Assemble move-conditional time logits from baseline + per-square components.

    time_logits(move) = baseline + time_from[from_sq] + time_to[to_sq]

    Args:
        time_baseline: (B, T) position-level baseline
        time_from_logits: (B, 64, T) per-square from-move time bias
        time_to_logits: (B, 64, T) per-square to-move time bias
        policy_target: (B,) move index (from*64 + to). May be the ignore_index
            sentinel (-100) for resign/flag samples — it is clamped to a valid
            square range here, and callers mask those samples out of the time
            loss via batch['has_move'].

    Returns:
        time_logits: (B, T) move-conditional time bin logits
    """
    # Extract from/to squares from policy target (clamp handles the -100 sentinel)
    clamped_target = policy_target.clamp(min=0, max=4095)
    from_sq = clamped_target // 64   # (B,)
    to_sq = clamped_target % 64      # (B,)

    B = time_baseline.shape[0]
    batch_idx = torch.arange(B, device=time_baseline.device)

    # Gather per-square biases for the actual move played
    from_bias = time_from_logits[batch_idx, from_sq]  # (B, T)
    to_bias = time_to_logits[batch_idx, to_sq]         # (B, T)

    return time_baseline + from_bias + to_bias   # (B, T)


def compute_losses(outputs, batch, policy_weight=1.0, wdl_weight=0.5, time_weight=0.5,
                   promo_weight=0.05, resign_weight=0.05, flag_weight=0.05,
                   resign_pos_weight=20.0, flag_pos_weight=50.0, elo_weights=None):
    """Compute losses for model outputs.

    Args:
        outputs: tuple of (move_logits [B, 4096], wdl_logits [B, 3],
                          resign_logit [B], flag_logit [B] or None,
                          time_baseline [B, T], time_from_logits [B, 64, T],
                          time_to_logits [B, 64, T],
                          (optional promo_logits))
        batch: dict with policy_target, has_move, resign_target, flag_target,
            y_val_cls, time_target_cls, legal_mask
        resign_pos_weight / flag_pos_weight: BCE positive-class upweighting to
            counter the heavy resign/flag class imbalance.
        elo_weights: optional (B,) tensor of per-sample weights based on ELO bin
    """
    if len(outputs) == 8:
        (move_logits, wdl_logits, resign_logit, flag_logit,
         time_baseline, time_from_logits, time_to_logits, promo_logits) = outputs
    else:
        (move_logits, wdl_logits, resign_logit, flag_logit,
         time_baseline, time_from_logits, time_to_logits) = outputs
        promo_logits = None

    # has_move is False for resign/flag samples (no move played) — they are
    # excluded from the policy/time losses and trained via the binary heads.
    has_move = batch['has_move'].float()  # (B,)
    has_move_sum = has_move.sum() + 1e-8

    # Policy loss: cross-entropy over 4096 moves. Resign/flag samples carry
    # policy_target=-100 and are dropped via ignore_index (per-sample loss 0).
    policy_target = batch['policy_target']  # [B] int64 indices
    per_sample_policy = F.cross_entropy(
        move_logits, policy_target, reduction='none', ignore_index=-100)
    if elo_weights is not None:
        policy_loss = (per_sample_policy * has_move * elo_weights).sum() / has_move_sum
    else:
        policy_loss = (per_sample_policy * has_move).sum() / has_move_sum

    # Promotion loss: cross-entropy on 4 classes (Q, R, B, N)
    # Compute for all samples to avoid data-dependent branching that breaks torch.compile
    if promo_logits is not None and 'promo_target' in batch and 'promo_file' in batch:
        promo_target = batch['promo_target']  # [B] 0=None, 1=Q, 2=R, 3=B, 4=N
        promo_file = batch['promo_file']      # [B] 0-7
        promo_type = (promo_target - 1).clamp(min=0)  # [B] 0=Q,1=R,2=B,3=N (clamp to handle promo_target=0)

        # Select logits for each sample's promo_file: (B, 8, 4) -> (B, 4)
        batch_indices = torch.arange(promo_logits.shape[0], device=promo_logits.device)
        selected_logits = promo_logits[batch_indices, promo_file, :]  # [B, 4]

        # Compute per-sample loss
        per_sample_loss = F.cross_entropy(selected_logits, promo_type, reduction='none')  # [B]

        # Mask: only count samples with actual promotions (promo_target > 0)
        promo_mask = (promo_target > 0).float()
        if elo_weights is not None:
            promo_loss = (per_sample_loss * promo_mask * elo_weights).sum() / (promo_mask.sum() + 1e-8)
        else:
            promo_loss = (per_sample_loss * promo_mask).sum() / (promo_mask.sum() + 1e-8)
    else:
        promo_loss = torch.tensor(0.0, device=move_logits.device)

    # WDL loss: cross-entropy on Win/Draw/Loss (0=loss, 1=draw, 2=win)
    if elo_weights is not None:
        wdl_loss = (F.cross_entropy(wdl_logits, batch['y_val_cls'], reduction='none') * elo_weights).mean()
    else:
        wdl_loss = F.cross_entropy(wdl_logits, batch['y_val_cls'])

    # Time loss: move-conditional cross-entropy on T bins
    # time_logits(move) = baseline + time_from[from_sq] + time_to[to_sq]
    # Only computed for samples with a move (has_move) and when the time head exists.
    if time_baseline is not None:
        time_logits = assemble_time_logits(
            time_baseline, time_from_logits, time_to_logits, policy_target
        )
        per_sample_time_loss = F.cross_entropy(time_logits, batch['time_target_cls'], reduction='none')
        if elo_weights is not None:
            time_loss = (per_sample_time_loss * has_move * elo_weights).sum() / has_move_sum
        else:
            time_loss = (per_sample_time_loss * has_move).sum() / has_move_sum
    else:
        time_loss = torch.tensor(0.0, device=move_logits.device)

    # Resign loss: BCE over all samples. pos_weight upweights the rare positive
    # class so the head does not collapse to "never resign".
    resign_pw = torch.tensor(resign_pos_weight, device=move_logits.device)
    per_sample_resign = F.binary_cross_entropy_with_logits(
        resign_logit, batch['resign_target'], pos_weight=resign_pw, reduction='none')
    if elo_weights is not None:
        resign_loss = (per_sample_resign * elo_weights).mean()
    else:
        resign_loss = per_sample_resign.mean()

    # Flag loss: BCE over all samples (None for no_time_context models).
    if flag_logit is not None:
        flag_pw = torch.tensor(flag_pos_weight, device=move_logits.device)
        per_sample_flag = F.binary_cross_entropy_with_logits(
            flag_logit, batch['flag_target'], pos_weight=flag_pw, reduction='none')
        if elo_weights is not None:
            flag_loss = (per_sample_flag * elo_weights).mean()
        else:
            flag_loss = per_sample_flag.mean()
    else:
        flag_loss = torch.tensor(0.0, device=move_logits.device)

    total = (policy_weight * policy_loss +
             wdl_weight * wdl_loss +
             time_weight * time_loss +
             promo_weight * promo_loss +
             resign_weight * resign_loss +
             flag_weight * flag_loss)

    return total, {
        'policy': policy_loss.detach(),
        'wdl': wdl_loss.detach(),
        'time': time_loss.detach(),
        'promo': promo_loss.detach(),
        'resign': resign_loss.detach(),
        'flag': flag_loss.detach(),
    }


def compute_metrics(outputs, batch) -> Dict[str, torch.Tensor]:
    """Compute accuracy metrics."""
    if len(outputs) == 8:
        (move_logits, wdl_logits, resign_logit, flag_logit,
         time_baseline, time_from_logits, time_to_logits, _) = outputs
    else:
        (move_logits, wdl_logits, resign_logit, flag_logit,
         time_baseline, time_from_logits, time_to_logits) = outputs
    policy_target = batch['policy_target']

    # Policy accuracy (top-1 and top-5), over samples with a real move only
    # (resign/flag samples have policy_target=-100 and no move output).
    # Note: model already masks illegal moves to -inf, so this is effectively legal_top1
    hm = batch['has_move'].float()
    hm_sum = hm.sum() + 1e-8
    top1 = ((move_logits.argmax(dim=1) == policy_target).float() * hm).sum() / hm_sum
    top5_preds = move_logits.topk(5, dim=1).indices  # [B, 5]
    top5_hit = (top5_preds == policy_target.unsqueeze(1)).any(dim=1).float()
    top5 = (top5_hit * hm).sum() / hm_sum

    # WDL classification accuracy
    wdl_acc = (wdl_logits.argmax(dim=1) == batch['y_val_cls']).float().mean()

    # Resign / flag binary metrics. Accuracy is a sanity check only — under the
    # heavy class imbalance it is trivially near-100%; recall (TP / actual
    # positives) is the meaningful signal.
    resign_target = batch['resign_target']
    resign_pred = (resign_logit > 0).float()
    resign_acc = (resign_pred == resign_target).float().mean()
    resign_recall = (resign_pred * resign_target).sum() / (resign_target.sum() + 1e-8)
    if flag_logit is not None:
        flag_target = batch['flag_target']
        flag_pred = (flag_logit > 0).float()
        flag_acc = (flag_pred == flag_target).float().mean()
        flag_recall = (flag_pred * flag_target).sum() / (flag_target.sum() + 1e-8)
    else:
        flag_acc = torch.tensor(0.0, device=move_logits.device)
        flag_recall = torch.tensor(0.0, device=move_logits.device)

    # Time classification accuracy (move-conditional, T bins)
    if time_baseline is not None:
        time_logits = assemble_time_logits(
            time_baseline, time_from_logits, time_to_logits, policy_target
        )
        time_cls_acc = (time_logits.argmax(dim=1) == batch['time_target_cls']).float().mean()
    else:
        time_logits = None
        time_cls_acc = torch.tensor(0.0, device=move_logits.device)

    # Time top-5 accuracy and close accuracy (within 2 bins = within 2 seconds, raw-second bins)
    if time_logits is not None:
        time_top5_preds = time_logits.topk(5, dim=1).indices
        time_top5 = (time_top5_preds == batch['time_target_cls'].unsqueeze(1)).any(dim=1).float().mean()
        time_pred_bin = time_logits.argmax(dim=1)
        time_close = (torch.abs(time_pred_bin - batch['time_target_cls']) <= 2).float().mean()
    else:
        time_top5 = torch.tensor(0.0, device=move_logits.device)
        time_close = torch.tensor(0.0, device=move_logits.device)

    # Promotion accuracy - no data-dependent branching
    promo_acc = torch.tensor(0.0, device=move_logits.device)
    promo_count = torch.tensor(0.0, device=move_logits.device)

    if 'promo_target' in batch and len(outputs) == 8:
        promo_logits = outputs[7]
        promo_target = batch['promo_target']
        promo_file = batch['promo_file']
        promo_type = (promo_target - 1).clamp(min=0)  # Handle promo_target=0

        promo_mask = (promo_target > 0).float()
        promo_count = promo_mask.sum()

        # Select logits for each sample's promo_file: (B, 8, 4) -> (B, 4)
        batch_indices = torch.arange(promo_logits.shape[0], device=promo_logits.device)
        selected_logits = promo_logits[batch_indices, promo_file, :]

        promo_preds = selected_logits.argmax(dim=1)
        promo_correct = (promo_preds == promo_type).float()
        promo_acc = (promo_correct * promo_mask).sum() / (promo_mask.sum() + 1e-8)

    return {
        'policy_top1': top1.detach(),
        'policy_top5': top5.detach(),
        'wdl_acc': wdl_acc.detach(),
        'resign_acc': resign_acc.detach(),
        'resign_recall': resign_recall.detach(),
        'flag_acc': flag_acc.detach(),
        'flag_recall': flag_recall.detach(),
        'time_cls_acc': time_cls_acc.detach(),
        'time_top5': time_top5.detach(),
        'time_close': time_close.detach(),
        'promo_acc': promo_acc.detach(),
        'promo_count': promo_count.detach(),
    }


# ELO bins: 1200-1300, 1300-1400, ..., 2500-2600
ELO_BINS = list(range(1200, 2600, 100))  # [1200, 1300, ..., 2500]
ELO_BIN_NAMES = [f"{lo}-{lo+100}" for lo in ELO_BINS]

# Time-control categories from process_pgn.py (Lichess perfType).
# Index matches the tc_cat int stored in the parquet.
TC_NAMES = ["bullet", "blitz", "rapid", "classical", "ultraBullet"]
NUM_TC = len(TC_NAMES)

# Game-phase buckets by move_ply (half-move counter).
# Bucket index = 0 if ply < 20, 1 if 20 ≤ ply < 60, 2 if ply ≥ 60.
# Standard chess heuristic — opening is the high-entropy phase where multiple
# moves are equally reasonable (theory still active); middlegame is where
# tactical structure commits and top-1 is most learnable; endgame has fewer
# pieces / more forced sequences. Defined here as constants so they're easy
# to tweak without hunting through code.
PHASE_NAMES = ["opening", "middlegame", "endgame"]
PHASE_BOUNDS = (20, 60)  # exclusive upper bound of bucket 0, bucket 1
NUM_PHASES = len(PHASE_NAMES)


def _phase_indices(move_ply: torch.Tensor) -> torch.Tensor:
    """Map (B,) move_ply tensor → (B,) long tensor in {0=opening, 1=middlegame, 2=endgame}."""
    lo, hi = PHASE_BOUNDS
    idx = torch.zeros_like(move_ply, dtype=torch.long)
    idx = idx + (move_ply >= lo).long()    # promote ≥lo to ≥1
    idx = idx + (move_ply >= hi).long()    # promote ≥hi to 2
    return idx


def evaluate(model, loader, device, non_blocking, max_batches=0, args=None) -> Dict[str, float]:
    """Evaluate model on validation data with per-ELO accuracy tracking."""
    model.eval()
    totals = defaultdict(float)
    sample_count = 0

    # Per-ELO bin accumulators
    num_bins = len(ELO_BINS)
    elo_top1_correct = torch.zeros(num_bins, device=device)
    elo_top5_correct = torch.zeros(num_bins, device=device)
    elo_counts = torch.zeros(num_bins, device=device)

    # Per-TC accumulators (bullet/blitz/rapid/classical/ultraBullet)
    tc_top1_correct = torch.zeros(NUM_TC, device=device)
    tc_top5_correct = torch.zeros(NUM_TC, device=device)
    tc_counts = torch.zeros(NUM_TC, device=device)
    # Per-TC time-head CE (over has_move samples only) — isolates whether the
    # time signal (e.g. tempo) helps move-time prediction in fast TCs specifically,
    # which the aggregate time-CE dilutes.
    tc_time_loss_sum = torch.zeros(NUM_TC, device=device)
    tc_time_counts = torch.zeros(NUM_TC, device=device)

    # Per-phase accumulators (opening/middlegame/endgame)
    phase_top1_correct = torch.zeros(NUM_PHASES, device=device)
    phase_top5_correct = torch.zeros(NUM_PHASES, device=device)
    phase_counts = torch.zeros(NUM_PHASES, device=device)

    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device, non_blocking)

            outputs = model(batch, return_promo=True)
            loss, loss_terms = compute_losses(
                outputs, batch,
                policy_weight=args.policy_weight if args else 1.0,
                wdl_weight=args.wdl_weight if args else 1.0,
                time_weight=args.time_weight if args else 0.5,
                promo_weight=args.promo_weight if args else 0.1,
                resign_weight=args.resign_weight if args else 0.1,
                flag_weight=args.flag_weight if args else 0.1,
                resign_pos_weight=args.resign_pos_weight if args else 20.0,
                flag_pos_weight=args.flag_pos_weight if args else 50.0,
            )
            metrics = compute_metrics(outputs, batch)

            batch_size = batch['board_history'].size(0)
            sample_count += batch_size
            totals['loss'] += loss.detach() * batch_size
            for name, value in {**loss_terms, **metrics}.items():
                totals[name] += value * batch_size

            # Per-ELO accuracy tracking
            move_logits = outputs[0]
            policy_target = batch['policy_target']

            # Get active_elo from scalars (index 0, normalized as (elo-1900)/700)
            # Denormalize: elo = scalars[:, 0] * 700 + 1900
            active_elo = batch['scalars'][:, 0] * 700 + 1900

            # Compute per-sample correctness
            top1_correct = (move_logits.argmax(dim=1) == policy_target).float()
            top5_correct = (move_logits.topk(5, dim=1).indices == policy_target.unsqueeze(1)).any(dim=1).float()

            # Bin the ELOs: clamp to [1200, 2599] then bin
            elo_clamped = active_elo.clamp(1200, 2599)
            bin_indices = ((elo_clamped - 1200) / 100).long().clamp(0, num_bins - 1)

            # Accumulate using scatter_add
            elo_top1_correct.scatter_add_(0, bin_indices, top1_correct)
            elo_top5_correct.scatter_add_(0, bin_indices, top5_correct)
            elo_counts.scatter_add_(0, bin_indices, torch.ones_like(top1_correct))

            # Per-TC accumulators (same pattern). Skip if tc_cat absent (older shards).
            if 'tc_cat' in batch:
                tc_cat = batch['tc_cat'].clamp(0, NUM_TC - 1)
                tc_top1_correct.scatter_add_(0, tc_cat, top1_correct)
                tc_top5_correct.scatter_add_(0, tc_cat, top5_correct)
                tc_counts.scatter_add_(0, tc_cat, torch.ones_like(top1_correct))

                # Per-TC time-head CE (has_move samples only). Mirrors the
                # training time loss: logits = baseline + from[from_sq] + to[to_sq].
                time_baseline = outputs[4]
                if time_baseline is not None and 'time_target_cls' in batch:
                    time_logits = assemble_time_logits(
                        time_baseline, outputs[5], outputs[6], policy_target
                    )
                    per_sample_time = F.cross_entropy(
                        time_logits, batch['time_target_cls'], reduction='none'
                    ).float()
                    has_move_f = batch['has_move'].float()
                    tc_time_loss_sum.scatter_add_(0, tc_cat, per_sample_time * has_move_f)
                    tc_time_counts.scatter_add_(0, tc_cat, has_move_f)

            # Per-phase accumulators (derived from scalars[:, 2]).
            move_ply = batch['scalars'][:, 2] * 100
            phase_idx = _phase_indices(move_ply)
            phase_top1_correct.scatter_add_(0, phase_idx, top1_correct)
            phase_top5_correct.scatter_add_(0, phase_idx, top5_correct)
            phase_counts.scatter_add_(0, phase_idx, torch.ones_like(top1_correct))

    results = {k: (v / sample_count).item() for k, v in totals.items()}

    # Add per-ELO metrics
    for i, bin_name in enumerate(ELO_BIN_NAMES):
        count = elo_counts[i].item()
        if count > 0:
            results[f"elo_{bin_name}_top1"] = (elo_top1_correct[i] / count).item()
            results[f"elo_{bin_name}_top5"] = (elo_top5_correct[i] / count).item()
            results[f"elo_{bin_name}_count"] = int(count)

    # Add per-TC metrics
    for i, tc_name in enumerate(TC_NAMES):
        count = tc_counts[i].item()
        if count > 0:
            results[f"tc_{tc_name}_top1"] = (tc_top1_correct[i] / count).item()
            results[f"tc_{tc_name}_top5"] = (tc_top5_correct[i] / count).item()
            results[f"tc_{tc_name}_count"] = int(count)
        tc_time_n = tc_time_counts[i].item()
        if tc_time_n > 0:
            results[f"tc_{tc_name}_timece"] = (tc_time_loss_sum[i] / tc_time_n).item()

    # In-range top1/top5 (1200 ≤ active_elo < 2500) — derived for free from the
    # per-ELO bin tallies, restricted to the 1200-2500 band the model is trained on.
    in_range_n = elo_counts[:13].sum().item()
    total_n = elo_counts.sum().item()
    if in_range_n > 0:
        results["policy_top1_inrange"] = (elo_top1_correct[:13].sum() / in_range_n).item()
        results["policy_top5_inrange"] = (elo_top5_correct[:13].sum() / in_range_n).item()
    if total_n > 0:
        results["in_range_frac"] = in_range_n / total_n

    # Per-phase metrics (opening/middlegame/endgame). Reveals whether the
    # dataset's move_ply distribution is dragging or boosting the global mean.
    for i, phase in enumerate(PHASE_NAMES):
        count = phase_counts[i].item()
        if count > 0:
            results[f"phase_{phase}_top1"] = (phase_top1_correct[i] / count).item()
            results[f"phase_{phase}_top5"] = (phase_top5_correct[i] / count).item()
            results[f"phase_{phase}_count"] = int(count)

    return results


def compute_elo_sample_weights(scalars, elo_bin_weights):
    """Look up per-sample loss weights from ELO bin.

    Args:
        scalars: (B, ...) tensor, scalars[:, 0] is normalized active ELO
        elo_bin_weights: (13,) tensor of per-bin weights (already normalized to mean=1.0)
    Returns:
        (B,) tensor of per-sample weights
    """
    active_elo = scalars[:, 0] * 700 + 1900
    bins = ((active_elo.clamp(1200, 2499) - 1200) / 100).long().clamp(0, 12)
    return elo_bin_weights[bins]


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    args,
    non_blocking: bool,
    profiler=None,
    val_loader=None,
    best_val_loss: float = float("inf"),
    run_dir: Path = None,
    resume_samples: int = 0,
    resume_micro_step: int = 0,
    general_loader=None,
    elo_bin_weights=None,
):
    model.train()
    totals = defaultdict(float)
    sample_count = 0
    samples_this_epoch = 0  # Track for checkpointing
    log_timer = time.perf_counter()
    last_log_step = global_step
    last_log_sample_count = 0
    micro_step = resume_micro_step  # Resume from saved micro_step
    optimizer.zero_grad(set_to_none=True)

    # Per-TC windowed accumulators (reset every log_interval). One-batch values
    # are noisy because rare TCs see few samples per batch; windowing smooths.
    tc_top1_window = torch.zeros(NUM_TC, device=device)
    tc_counts_window = torch.zeros(NUM_TC, device=device)

    # In-range (1200 ≤ active_elo < 2500) windowed accumulators. Compares the
    # the run against the 1200-2500 band it was trained and evaluated on;
    # the new dataset includes OOR samples that drag the global metric down.
    in_range_top1_window = 0.0
    in_range_count_window = 0
    total_count_window = 0  # for logging in-range fraction over the same window

    # Per-game-phase (opening / middlegame / endgame) windowed accumulators.
    # Diagnostic for the move_ply distribution shift between datasets — the new
    # dataset samples ~33% earlier in games on average, and opening positions
    # are intrinsically harder to top-1 predict than middlegame.
    phase_top1_window = torch.zeros(NUM_PHASES, device=device)
    phase_counts_window = torch.zeros(NUM_PHASES, device=device)

    # For resume: skip samples already processed
    if resume_samples > 0:
        print(f"[train] Resuming epoch {epoch}: skipping {resume_samples:,} samples already processed")

    # --- General data mixing setup ---
    general_iter = None
    mix_ratio = getattr(args, 'mix_ratio', 0.0) if general_loader is not None else 0.0
    if general_loader is not None and mix_ratio > 0:
        general_iter = iter(general_loader)
        print(f"[train] Mixing {mix_ratio*100:.0f}% general data per epoch")

    for step, batch in enumerate(loader, start=1):
        # --- General data mixing: probabilistically replace batch with general data ---
        if general_iter is not None and mix_ratio > 0 and random.random() < mix_ratio:
            try:
                batch = next(general_iter)
            except StopIteration:
                # Restart general data iterator
                general_iter = iter(general_loader)
                try:
                    batch = next(general_iter)
                except StopIteration:
                    pass  # Fall through to player data if general loader is empty

        batch = move_batch_to_device(batch, device, non_blocking)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(batch, return_promo=True)
            elo_weights = compute_elo_sample_weights(batch['scalars'], elo_bin_weights) if elo_bin_weights is not None else None
            raw_loss, loss_terms = compute_losses(
                outputs, batch,
                policy_weight=args.policy_weight,
                wdl_weight=args.wdl_weight,
                time_weight=args.time_weight,
                promo_weight=args.promo_weight,
                resign_weight=args.resign_weight,
                flag_weight=args.flag_weight,
                resign_pos_weight=args.resign_pos_weight,
                flag_pos_weight=args.flag_pos_weight,
                elo_weights=elo_weights,
            )
            loss = raw_loss / args.grad_accum_steps

        scaler.scale(loss).backward()
        micro_step += 1

        should_step = micro_step % args.grad_accum_steps == 0
        if should_step:
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metrics = compute_metrics(outputs, batch)
        batch_size = batch['board_history'].size(0)
        sample_count += batch_size
        samples_this_epoch += batch_size  # Track total samples for checkpointing
        totals['loss'] += raw_loss.detach() * batch_size
        for name, value in {**loss_terms, **metrics}.items():
            totals[name] += value * batch_size

        # Per-TC + in-range windowed accumulation. Cheap (1 scatter_add + 1
        # mask) per batch; window-aggregated they're robust between log_intervals.
        with torch.no_grad():
            top1_correct = (outputs[0].argmax(dim=1) == batch['policy_target']).float()
            if 'tc_cat' in batch:
                tc_cat = batch['tc_cat'].clamp(0, NUM_TC - 1)
                tc_top1_window.scatter_add_(0, tc_cat, top1_correct)
                tc_counts_window.scatter_add_(0, tc_cat, torch.ones_like(top1_correct))
            # Recover active_elo from normalized scalars[:, 0].
            active_elo = batch['scalars'][:, 0] * 700 + 1900
            in_range = (active_elo >= 1200) & (active_elo < 2500)
            n_in_range = in_range.sum().item()
            if n_in_range > 0:
                in_range_top1_window += (top1_correct * in_range.float()).sum().item()
                in_range_count_window += n_in_range
            total_count_window += active_elo.numel()
            # Game phase from scalars[:, 2] (ply_norm = move_ply / 100).
            move_ply = batch['scalars'][:, 2] * 100
            phase_idx = _phase_indices(move_ply)
            phase_top1_window.scatter_add_(0, phase_idx, top1_correct)
            phase_counts_window.scatter_add_(0, phase_idx, torch.ones_like(top1_correct))

        if should_step:
            global_step += 1
            if global_step % args.log_interval == 0:
                writer.add_scalar("train/total_loss", raw_loss.item(), global_step)
                for name, value in loss_terms.items():
                    writer.add_scalar(f"train/{name}_loss", value.item(), global_step)
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", value.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                if len(optimizer.param_groups) > 1:
                    writer.add_scalar("train/lr_heads", optimizer.param_groups[1]["lr"], global_step)

                # Dead-encoder watch: log ||time_repr|| and ||elo_repr|| when the
                # broadcast-time variant is active. Trajectory is what matters
                # (slowly decaying toward 0 = dying), not any absolute ratio.
                # torch.compile wraps the model; the underlying module is on _orig_mod.
                base_model = getattr(model, "_orig_mod", model)
                time_norm = getattr(base_model, "_last_time_repr_norm", None)
                elo_norm = getattr(base_model, "_last_elo_repr_norm", None)
                if time_norm is not None:
                    writer.add_scalar("train/time_repr_norm", time_norm.item(), global_step)
                if elo_norm is not None:
                    writer.add_scalar("train/elo_repr_norm", elo_norm.item(), global_step)

                # Per-square state_proj layout (per-token elo fusion):
                #   [castling(4) | rep(history_len) | clock(2*history_len if per_pos)
                #    | elo(2*elo_concat_dim)]
                # Track L2 norms of the post-castling blocks so a silent failure of
                # any pathway (clocks staying at init, elo block never learning)
                # is visible alongside the existing dead-encoder watch.
                ie = getattr(base_model, "input_encoder", None)
                if ie is not None and hasattr(ie, "state_proj"):
                    w = ie.state_proj.weight  # (d_model, state_dim)
                    castle_dim = 4
                    rep_dim = ie.history_len
                    clock_dim = 2 * ie.history_len if ie.per_position_clocks else 0
                    elo_dim = 2 * ie.elo_concat_dim
                    cursor = castle_dim + rep_dim
                    if ie.per_position_clocks:
                        clock_block = w[:, cursor:cursor + clock_dim]
                        clock_norm = clock_block.detach().pow(2).sum().sqrt()
                        writer.add_scalar("train/state_proj_clock_block_norm", clock_norm.item(), global_step)
                        cursor += clock_dim
                    elo_block = w[:, cursor:cursor + elo_dim]
                    elo_block_norm = elo_block.detach().pow(2).sum().sqrt()
                    writer.add_scalar("train/state_proj_elo_block_norm", elo_block_norm.item(), global_step)

                # Per-TC train top1 over the window since the last log step.
                # Skip TCs with zero samples (rare ones — ultraBullet, classical
                # under bullet-heavy training mixes — would otherwise spam NaN).
                if tc_counts_window.sum().item() > 0:
                    for i, tc_name in enumerate(TC_NAMES):
                        c = tc_counts_window[i].item()
                        if c > 0:
                            acc = (tc_top1_window[i] / c).item()
                            writer.add_scalar(f"train/tc_{tc_name}_top1", acc, global_step)
                    tc_top1_window.zero_()
                    tc_counts_window.zero_()

                # In-range (1200 ≤ active_elo < 2500) top1 over the window —
                # an apples-to-apples comparison against runs that only trained
                # on 1200-2500. Also log the in-range fraction so you can see
                # how much OOR drag is in play.
                if in_range_count_window > 0:
                    writer.add_scalar("train/policy_top1_inrange",
                                      in_range_top1_window / in_range_count_window, global_step)
                if total_count_window > 0:
                    writer.add_scalar("train/in_range_frac",
                                      in_range_count_window / total_count_window, global_step)
                in_range_top1_window = 0.0
                in_range_count_window = 0
                total_count_window = 0

                # Per-phase train top1 over the same window.
                if phase_counts_window.sum().item() > 0:
                    for i, phase in enumerate(PHASE_NAMES):
                        c = phase_counts_window[i].item()
                        if c > 0:
                            writer.add_scalar(f"train/phase_{phase}_top1",
                                              (phase_top1_window[i] / c).item(), global_step)
                    phase_top1_window.zero_()
                    phase_counts_window.zero_()

                elapsed = max(1e-6, time.perf_counter() - log_timer)
                steps_since = max(1, global_step - last_log_step)
                samples_since = sample_count - last_log_sample_count
                it_per_sec = steps_since / elapsed
                samples_per_sec = samples_since / elapsed
                print(
                    f"[train] step {global_step:,} | samples {resume_samples + samples_this_epoch:,} | "
                    f"{it_per_sec:.2f} it/s ({samples_per_sec:,.0f} samp/s) | loss {raw_loss.item():.3f} | "
                    f"policy {loss_terms['policy'].item():.3f} | wdl {loss_terms['wdl'].item():.4f} | "
                    f"resign {loss_terms['resign'].item():.3f} | flag {loss_terms['flag'].item():.3f} | "
                    f"top1 {metrics['policy_top1'].item():.3f}"
                )
                log_timer = time.perf_counter()
                last_log_step = global_step
                last_log_sample_count = sample_count

            # Mid-epoch validation
            if args.val_interval > 0 and val_loader is not None and global_step % args.val_interval == 0:
                print(f"\n[val] Running validation at step {global_step:,}...")
                val_stats = evaluate(model, val_loader, device, non_blocking, max_batches=args.val_batches, args=args)
                print(
                    f"[val] step {global_step:,} | loss {val_stats.get('loss', 0):.4f} | "
                    f"top1 {val_stats.get('policy_top1', 0):.3f} | "
                    f"wdl {val_stats.get('wdl_acc', 0):.3f} | time_cls {val_stats.get('time_cls_acc', 0):.3f}"
                )
                # Print per-ELO accuracy summary
                elo_keys = [k for k in val_stats if k.startswith("elo_") and k.endswith("_top1")]
                if elo_keys:
                    elo_summary = " | ".join(
                        f"{k.replace('elo_', '').replace('_top1', '')}:{val_stats[k]:.2f}"
                        for k in sorted(elo_keys)  # all bins (full table for the band diagnostic)
                    )
                    print(f"[val] per-ELO top1: {elo_summary}")
                # Print per-TC time-CE (fast→slow) — this is where a tempo effect shows.
                tce_order = ["ultraBullet", "bullet", "blitz", "rapid", "classical"]
                tce_summary = " | ".join(
                    f"{tc}:{val_stats[f'tc_{tc}_timece']:.3f}"
                    for tc in tce_order if f"tc_{tc}_timece" in val_stats
                )
                if tce_summary:
                    print(f"[val] per-TC time-CE: {tce_summary}")
                for name, value in val_stats.items():
                    writer.add_scalar(f"val/{name}", value, global_step)

                if val_stats['loss'] < best_val_loss:
                    best_val_loss = val_stats['loss']
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                        Path(args.checkpoint_dir) / f"chessformer_{args.run_slug}_best.pt",
                        args.config, run_dir,
                        samples_consumed=resume_samples + samples_this_epoch,
                        micro_step=micro_step % args.grad_accum_steps,
                        config_dict=getattr(args, "config_dict", None),
                    )

                model.train()
                log_timer = time.perf_counter()
                last_log_step = global_step

            # Periodic checkpoint for resume (overwrites previous)
            if args.checkpoint_interval > 0 and global_step % args.checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                    Path(args.checkpoint_dir) / f"chessformer_{args.run_slug}_latest.pt",
                    args.config, run_dir,
                    samples_consumed=resume_samples + samples_this_epoch,
                    micro_step=micro_step % args.grad_accum_steps,
                    config_dict=getattr(args, "config_dict", None),
                )

            # Periodic archival checkpoint (non-overwriting)
            if args.checkpoint_archive_interval > 0 and global_step % args.checkpoint_archive_interval == 0:
                archive_path = Path(args.checkpoint_archive_dir) / f"chessformer_{args.run_slug}_step{global_step}.pt"
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                    archive_path,
                    args.config, run_dir,
                    samples_consumed=resume_samples + samples_this_epoch,
                    micro_step=micro_step % args.grad_accum_steps,
                    config_dict=getattr(args, "config_dict", None),
                )

        if profiler is not None:
            profiler.step()

        # Bound diagnostic runs to an exact optimizer-step budget.
        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"[train] reached --max-steps {args.max_steps:,} — stopping epoch early")
            break

    # Handle remaining accumulated gradients
    if micro_step % args.grad_accum_steps != 0:
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    epoch_stats = {k: (v / sample_count).item() for k, v in totals.items()}
    return epoch_stats, global_step, best_val_loss, resume_samples + samples_this_epoch


def _load_state_dict_compile_aware(model, model_state):
    """Load a state_dict regardless of whether the saved-from / loaded-into
    model was wrapped by torch.compile (which prefixes keys with _orig_mod.).

    Strategy: normalize away the _orig_mod. prefix on the source side, then
    target the underlying (un-compiled) module directly. Works in both
    compile→non-compile and non-compile→compile directions.
    """
    normalized = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in model_state.items()
    }
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(normalized)


def save_checkpoint(
    model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
    path: Path, config_name: str, run_dir: Optional[Path] = None,
    samples_consumed: int = 0, micro_step: int = 0,
    config_dict: Optional[dict] = None,
):
    """Save checkpoint with dataset position for seamless resume.

    Args:
        config_name: Short config slug ("tiny", "small", etc.) for back-compat.
        config_dict: Full resolved model config (architecture flags included).
            When present, inference tooling can reconstruct the exact model
            architecture from the checkpoint alone.
        samples_consumed: Number of samples processed in current epoch (for resume)
        micro_step: Current micro-step within gradient accumulation (for resume)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": config_name,
        "version": "2.0",  # Version bump for new resume fields
        # Resume state
        "samples_consumed": samples_consumed,
        "micro_step": micro_step,
    }
    if config_dict is not None:
        checkpoint["config_dict"] = dict(config_dict)
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if run_dir is not None:
        checkpoint["run_dir"] = str(run_dir)
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path} (epoch={epoch}, step={global_step}, samples={samples_consumed:,})")


def load_checkpoint(path: Path, model, optimizer, scheduler=None, scaler=None, device=None):
    """Load checkpoint and return resume state.
    
    Returns:
        Tuple of (epoch, global_step, best_val_loss, run_dir, samples_consumed, micro_step)
        - samples_consumed: samples processed in the saved epoch (for resume mid-epoch)
        - micro_step: gradient accumulation step (for resume)
    """
    print(f"[checkpoint] Loading from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_state = checkpoint["model"]
    _load_state_dict_compile_aware(model, model_state)

    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    run_dir = checkpoint.get("run_dir", None)

    # Resume state (new in v2.0)
    samples_consumed = checkpoint.get("samples_consumed", 0)
    micro_step = checkpoint.get("micro_step", 0)

    print(f"[checkpoint] Resumed from epoch {epoch}, step {global_step}, best_val_loss {best_val_loss:.4f}")
    print(f"[checkpoint] Resume state: samples_consumed={samples_consumed:,}, micro_step={micro_step}")
    return epoch, global_step, best_val_loss, run_dir, samples_consumed, micro_step


def load_pretrained_weights(path: Path, model, device=None):
    """Load only model weights from a checkpoint (for fine-tuning).
    
    Unlike load_checkpoint, this does NOT load optimizer, scheduler, or scaler state.
    The model starts training fresh with a new optimizer, but initialized from
    pretrained weights.
    """
    print(f"[finetune] Loading pretrained weights from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_state = checkpoint["model"]
    _load_state_dict_compile_aware(model, model_state)

    config_name = checkpoint.get("config", "unknown")
    # Backwards compat: strip legacy "-v2" suffix from old checkpoints
    config_name = config_name.replace("-v2", "")
    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("global_step", "?")
    best_loss = checkpoint.get("best_val_loss", "?")
    print(f"[finetune] Loaded {config_name} weights (epoch={epoch}, step={step}, best_val_loss={best_loss})")
    print("[finetune] Optimizer/scheduler state NOT loaded (fresh training start)")


# Head parameter name patterns — these get the higher learning rate in finetune mode
_HEAD_PATTERNS = (
    "policy_query", "policy_key", "policy_scale",
    "promo_bias_proj",
    "wdl_head",
    "resign_head", "flag_head",
    "time_baseline_pool", "time_baseline_head", "time_from_proj", "time_to_proj",
)


def build_finetune_param_groups(model, lr_backbone: float, lr_heads: float,
                                 weight_decay: float, freeze_layers: int = 0):
    """Build parameter groups with discriminative learning rates for fine-tuning.
    
    Splits model parameters into:
      - backbone (embeddings + transformer layers): lower LR
      - heads (policy, value, time, promo, etc.): higher LR
    
    Optionally freezes the first N transformer layers entirely.
    
    Returns:
        List of param group dicts for the optimizer.
    """
    backbone_params = []
    head_params = []
    frozen_count = 0

    for name, param in model.named_parameters():
        # Check if this parameter belongs to a frozen layer
        if freeze_layers > 0:
            # Match patterns like "layers.0.", "layers.1.", etc.
            # Also handle _orig_mod prefix from torch.compile
            clean_name = name.replace("_orig_mod.", "")
            is_frozen = False
            for i in range(freeze_layers):
                if f"layers.{i}." in clean_name:
                    is_frozen = True
                    break
            if is_frozen:
                param.requires_grad = False
                frozen_count += 1
                continue

        # Classify as head or backbone
        is_head = any(pattern in name for pattern in _HEAD_PATTERNS)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if frozen_count > 0:
        print(f"[finetune] Frozen {frozen_count} parameters in first {freeze_layers} transformer layers")

    trainable_backbone = sum(p.numel() for p in backbone_params)
    trainable_heads = sum(p.numel() for p in head_params)
    total = trainable_backbone + trainable_heads
    print("[finetune] Parameter groups:")
    print(f"  Backbone: {trainable_backbone:,} params (lr={lr_backbone})")
    print(f"  Heads:    {trainable_heads:,} params (lr={lr_heads})")
    print(f"  Total trainable: {total:,}")

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_heads, "weight_decay": weight_decay},
    ]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_slug(args, config: dict) -> str:
    """Build a TensorBoard run slug encoding the ablation axes.

    Format: <config>_elo-<variant>_time-<variant>[-<featurize>]_s<seed>[_d<data_seed>][_det][_ft]

    Examples:
      5m_elo-endpoints_time-tokens_tempotok_s67
      5m_elo-endpoints_time-tokens_tempotok_gabrot_s67
      23m_elo-endpoints_time-mlp-log1p_s67_det
      23m_elo-endpoints_time-tokens_tempotok_s68_d43   (seed sweep: --seed 68 --data-seed 43)

    The d<data_seed> tag is only appended when --data-seed differs from the
    dataset.py default (42), so pre-existing single-seed runs keep their slug.

    Order is fixed (config → elo → time → seed → flags) so runs sort/diff
    cleanly across an ablation sweep.
    """
    # "elo-endpoints" is a literal: Maia3EloEncoder (two anchors at Elo 0/5000)
    # is the only ELO encoder. Kept in the slug so re-runs reproduce the
    # checkpoint names used throughout the paper and the release manifest.
    parts = [args.config, "elo-endpoints"]
    if config.get("no_time_conditioning", False):
        # Time-blind ablation: the time-encoder/tempo/per-position tags would be
        # inert and misleading (all those inputs are zeroed), so collapse them
        # to a single unambiguous marker.
        parts.append("time-blind")
    elif not config.get("no_time_context", False):
        if args.time_encoder == "mlp":
            parts.append(f"time-{args.time_encoder}-{args.time_featurize}")
        elif args.time_encoder == "interp_channels":
            _d = args.time_interp_channel_dim or config["dim_vit"]
            parts.append(f"time-{args.time_encoder}-d{_d}")
        else:
            parts.append(f"time-{args.time_encoder}")
        if args.per_position_clocks:
            parts.append("pc-kg" if args.keep_global_time_encoder else "pc")
            if args.per_position_time_deltas:
                parts.append("dt")
        if args.tempo_history_token:
            parts.append("tempotok")
        if args.zero_time_history:
            parts.append("notempo")
    # GAB Q/K rotation ablation tag (omitted for the additive-only baseline).
    if args.gab_qk_rotation and args.gab_additive_bias:
        parts.append("gabrot")
    elif args.gab_qk_rotation and not args.gab_additive_bias:
        parts.append("gabrot-only")
    parts.append(f"s{args.seed}")
    if args.data_seed != 42:
        parts.append(f"d{args.data_seed}")
    if args.deterministic:
        parts.append("det")
    if args.finetune:
        parts.append("ft")
    if args.train_tc:
        parts.append(f"only-{args.train_tc}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    maia3_configs = {
        "3m": CONFIG_MAIA3_3M,
        "5m": CONFIG_MAIA3_5M,
        "23m": CONFIG_MAIA3_23M,
        "79m": CONFIG_MAIA3_79M,
    }
    config = copy.deepcopy(maia3_configs[args.config])

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        config['gradient_checkpointing'] = True

    # Apply dropout override for fine-tuning (modifies config before model creation)
    if args.dropout_override is not None:
        original_dropout = config.get('dropout', 0.1)
        config['dropout'] = args.dropout_override
        print(f"[config] Dropout override: {original_dropout} -> {args.dropout_override}")

    # Parse per-ELO loss weights
    elo_bin_weights = None
    if args.elo_loss_weights is not None:
        raw_weights = [float(x.strip()) for x in args.elo_loss_weights.split(",")]
        if len(raw_weights) != 13:
            raise ValueError(f"--elo-loss-weights must have 13 values (got {len(raw_weights)})")
        elo_bin_weights = torch.tensor(raw_weights, dtype=torch.float32)
        elo_bin_weights = elo_bin_weights / elo_bin_weights.mean()  # normalize to mean=1.0
        bin_names = [f"{lo}-{lo+100}" for lo in range(1200, 2600, 100)]
        print("[config] Per-ELO loss weights (normalized to mean=1.0):")
        for name, w in zip(bin_names, elo_bin_weights.tolist()):
            print(f"  {name}: {w:.3f}")

    device = torch.device(args.device)
    configure_precision(device, args.disable_tf32, deterministic=args.deterministic)

    # Thread the time-encoder variant + featurize sub-axis into the model config.
    # `no_time_context` configs skip time conditioning entirely; for those, the
    # variant is irrelevant (model never instantiates the encoder).
    if not config.get("no_time_context", False):
        config["time_encoder"] = args.time_encoder
        config["time_featurize"] = args.time_featurize
        config["per_position_clocks"] = bool(args.per_position_clocks)
        config["shrink_global_time"] = not bool(args.keep_global_time_encoder)
        config["per_position_time_deltas"] = bool(args.per_position_time_deltas)
        config["tempo_history_token"] = bool(args.tempo_history_token)
        config["zero_time_history"] = bool(args.zero_time_history)
        config["no_time_conditioning"] = bool(args.no_time_conditioning)
        if args.time_interp_channel_dim is not None:
            config["time_interp_channel_dim"] = int(args.time_interp_channel_dim)
        if args.time_encoder == "mlp":
            print(f"[config] time_encoder = {args.time_encoder} (featurize={args.time_featurize})")
        elif args.time_encoder == "interp_channels":
            _d = args.time_interp_channel_dim or config["dim_vit"]
            print(f"[config] time_encoder = {args.time_encoder} "
                  f"(per-signal dim={_d}{'=dim_vit' if not args.time_interp_channel_dim else ''})")
        else:
            print(f"[config] time_encoder = {args.time_encoder}")
        if config["per_position_clocks"]:
            if config["shrink_global_time"]:
                print("[config] per_position_clocks = True, global encoder = {base_time, increment} (shrunk)")
            else:
                print("[config] per_position_clocks = True, global encoder = full 6 inputs (keep_global_time_encoder)")
            print(f"[config] per_position_time_deltas = {config['per_position_time_deltas']} "
                  f"({'per-move deltas (tempo)' if config['per_position_time_deltas'] else 'absolute clocks'})")
        if config.get("tempo_history_token", False):
            _unit = "channel" if args.time_encoder == "interp_channels" else "token"
            print(f"[config] tempo_history_token = True (8-ply tempo {_unit} replaces my_last/opp_last)")
        if config.get("zero_time_history", False):
            print("[config] zero_time_history = True (ABLATION: tempo/move-time history blanked)")
        if config.get("no_time_conditioning", False):
            print("[config] no_time_conditioning = True (ABLATION: time-blind — full architecture, "
                  "constant zero time inputs, time + flag heads kept)")

    # ELO conditioning is always active and always Maia3EloEncoder: player and
    # opponent rating interpolate between two learned anchors at Elo 0 and 5000.
    config["elo_encoder"] = "endpoints"

    # GAB Q/K rotation ablation knobs.
    config["gab_qk_rotation"] = bool(args.gab_qk_rotation)
    config["gab_additive_bias"] = bool(args.gab_additive_bias)
    if not config["gab_additive_bias"] and not config["gab_qk_rotation"]:
        raise ValueError("--no-gab-additive-bias requires --gab-qk-rotation "
                         "(GAB must emit at least one signal)")
    _arm = ("A (additive-only)" if not config["gab_qk_rotation"]
            else "B (additive + rotation)" if config["gab_additive_bias"]
            else "C (rotation-only)")
    print(f"[config] GAB arm = {_arm}  "
          f"(additive_bias={config['gab_additive_bias']}, qk_rotation={config['gab_qk_rotation']})")

    # Stash the resolved config dict on args so save_checkpoint call sites can
    # persist it into the .pt without threading another arg through everywhere.
    args.config_dict = config

    if elo_bin_weights is not None:
        elo_bin_weights = elo_bin_weights.to(device)

    model = Maia3Net(config).to(device=device)
    if args.compile_model:
        try:
            # Use 'default' mode when gradient accumulation is enabled to avoid instability
            # max-autotune can cause issues with gradient accumulation
            compile_mode = args.compile_mode
            if args.grad_accum_steps > 1 and compile_mode == "max-autotune":
                compile_mode = "default"
                print("[train] Switching compile mode to 'default' for gradient accumulation stability")
            model = torch.compile(model, mode=compile_mode)
            print(f"[train] torch.compile enabled with mode={compile_mode}")
        except Exception as exc:
            print(f"[train] torch.compile failed ({exc}); continuing without.")
    print(f"Using {args.config} config ({count_parameters(model):,} params, fp16 mixed precision)")

    # --- Fine-tune: load pretrained weights BEFORE creating optimizer ---
    if args.finetune:
        pretrained_path = Path(args.pretrained) if args.pretrained else (Path(args.resume) if args.resume else None)
        if pretrained_path is None:
            raise ValueError("Fine-tune mode requires --pretrained or --resume to specify the pretrained checkpoint")
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        load_pretrained_weights(pretrained_path, model, device)

        # Freeze layers if requested
        # Build param groups with discriminative LRs
        param_groups = build_finetune_param_groups(
            model, args.lr_backbone, args.lr_heads, args.weight_decay, args.freeze_layers
        )
        # Re-count after freezing
        print(f"[finetune] Trainable parameters after freezing: {count_parameters(model):,}")

    # Create dataloaders
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    train_filter_tc = TC_NAMES.index(args.train_tc) if args.train_tc else None
    val_filter_tc = TC_NAMES.index(args.val_tc) if args.val_tc else None
    if args.train_tc:
        print(f"[train] filtering to tc={args.train_tc} (id={train_filter_tc}) — diagnostic mode")
    if args.val_tc:
        print(f"[val]   filtering to tc={args.val_tc} (id={val_filter_tc})")
    train_loader = create_dataloader(
        args.data_dir, split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.data_seed,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        filter_tc=train_filter_tc,
        history_mask_prob=args.history_mask_prob,
    )
    # When a held-out val directory is given, take ALL shards there with no
    # hash split (the 'all' mode). Previously this path used split='train'
    # which still applied a deterministic hash split and silently dropped
    # ~val_fraction of the shards.
    val_data_source = args.val_data_dir if args.val_data_dir else args.data_dir
    val_split_mode = 'all' if args.val_data_dir else 'val'
    val_loader = create_dataloader(
        val_data_source, split=val_split_mode,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        seed=args.data_seed,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        max_val_files=args.num_val_files,
        filter_tc=val_filter_tc,
    )

    # Optional: create general data loader for mixing (anti-catastrophic-forgetting)
    general_loader = None
    if args.mix_general_data:
        general_data_path = Path(args.mix_general_data)
        if not general_data_path.exists():
            print(f"WARNING: --mix-general-data path {general_data_path} does not exist, skipping mixing")
        else:
            general_loader = create_dataloader(
                str(general_data_path), split='train',
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.num_workers // 4),
                seed=args.data_seed,
                pin_memory=pin_memory,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                history_mask_prob=args.history_mask_prob,
            )
            print(f"[finetune] General data mixing enabled: {args.mix_ratio*100:.0f}% from {general_data_path}")

    train_samples = count_samples(args.data_dir, 'train', max_val_files=args.num_val_files)
    if args.val_data_dir:
        val_samples = count_samples(args.val_data_dir, 'all')
    else:
        val_samples = count_samples(args.data_dir, 'val', max_val_files=args.num_val_files)
    print(f"[dataset] Train samples: ~{train_samples:,} | Val samples: ~{val_samples:,}")

    # Optimizer
    optimizer_kwargs = {"weight_decay": args.weight_decay}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True

    if args.finetune:
        # Fine-tune: use discriminative LR param groups (lr is set per group)
        try:
            optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
    else:
        # Normal training: single LR for all params
        optimizer_kwargs["lr"] = args.lr
        try:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    # Scheduler
    approx_batches = max(1, math.ceil(train_samples / args.batch_size))
    steps_per_epoch = max(1, math.ceil(approx_batches / args.grad_accum_steps))
    if args.lr_total_steps is not None:
        total_steps = args.lr_total_steps
        print(f"[scheduler] total_steps overridden via --lr-total-steps: {total_steps:,} "
              f"(recomputed value would have been {args.epochs * steps_per_epoch:,})")
    else:
        total_steps = args.epochs * steps_per_epoch

    # LR Schedule
    warmup_steps = args.warmup_steps

    if args.cyclic_lr:
        # Cyclic cosine annealed schedule: oscillates between lr and min_lr
        # Each cycle is a full cosine period (lr -> min_lr -> lr)
        min_lr_ratio = args.min_lr / max(args.lr, 1e-9)
        cycle_steps = max(1, total_steps / args.cyclic_num_cycles)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # Cosine oscillation: ranges from 1.0 (peak) to min_lr_ratio (trough)
            progress = (step - warmup_steps) / cycle_steps
            cosine = 0.5 * (1 + math.cos(2 * math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"[scheduler] Cyclic cosine LR: {args.lr} <-> {args.min_lr}, "
              f"{args.cyclic_num_cycles} cycles over {total_steps:,} steps "
              f"(~{cycle_steps:.0f} steps/cycle)")
    else:
        min_lr_ratio = args.min_lr / max(args.lr, 1e-9)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # Cosine decay after warmup, floored at min_lr
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"[scheduler] Linear warmup ({warmup_steps} steps) + cosine decay to {args.min_lr:.2e}")

    if args.finetune:
        print(f"[scheduler] LR backbone: {args.lr_backbone} -> 0 | LR heads: {args.lr_heads} -> 0")

    # Mixed precision: use GradScaler for fp16 training
    scaler = GradScaler(enabled=device.type == "cuda")

    # Resume state
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    resumed_run_dir = None
    resume_samples = 0  # Samples already processed in current epoch
    resume_micro_step = 0  # Gradient accumulation step

    if args.resume and not args.finetune:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, global_step, best_val_loss, resumed_run_dir, resume_samples, resume_micro_step = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        # If we have samples_consumed > 0, we're resuming mid-epoch
        # Otherwise, start the next epoch
        if resume_samples == 0:
            start_epoch += 1
        else:
            print(f"[resume] Resuming mid-epoch {start_epoch} at sample {resume_samples:,}")

    # Build the ablation-aware run slug once; reuse for TB run dir and all
    # checkpoint filenames so concurrent ablation variants don't clobber
    # each other's best/latest/epoch files.
    args.run_slug = _run_slug(args, config)

    # Run directory
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
    elif resumed_run_dir and args.resume:
        run_dir = Path(resumed_run_dir)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.log_dir) / f"chessformer_{args.run_slug}_{timestamp}"
        print(f"[run] {run_dir}")

    writer = SummaryWriter(run_dir)
    if start_epoch == 1:
        writer.add_text("config", str(config))

    profiler_context = create_profiler(args, run_dir, device, steps_per_epoch)
    with profiler_context as profiler:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            # Ensure streaming parquet order changes each epoch (deterministically).
            if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)

            # Set skip samples for resume (only for the first epoch when resuming mid-epoch)
            if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_skip_samples"):
                train_loader.dataset.set_skip_samples(resume_samples)

            train_stats, global_step, best_val_loss, samples_consumed = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, writer,
                epoch, global_step, args, non_blocking, profiler,
                val_loader=val_loader, best_val_loss=best_val_loss, run_dir=run_dir,
                resume_samples=resume_samples, resume_micro_step=resume_micro_step,
                general_loader=general_loader, elo_bin_weights=elo_bin_weights,
            )

            # Stop the whole run once the step budget is hit (diagnostic mode).
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            # Reset resume state after first epoch (subsequent epochs start fresh)
            resume_samples = 0
            resume_micro_step = 0

            print(
                f"Train loss: {train_stats.get('loss', 0):.4f} | "
                f"top1 {train_stats.get('policy_top1', 0):.3f} | "
                f"wdl {train_stats.get('wdl_acc', 0):.3f} | "
                f"time_cls {train_stats.get('time_cls_acc', 0):.3f}"
            )

            val_stats = evaluate(model, val_loader, device, non_blocking, max_batches=args.val_batches, args=args)
            print(
                f"Val loss: {val_stats.get('loss', 0):.4f} | "
                f"top1 {val_stats.get('policy_top1', 0):.3f} | "
                f"wdl {val_stats.get('wdl_acc', 0):.3f} | "
                f"time_cls {val_stats.get('time_cls_acc', 0):.3f} | time_close {val_stats.get('time_close', 0):.3f}"
            )
            # Print per-ELO accuracy breakdown
            elo_keys = sorted([k for k in val_stats if k.startswith("elo_") and k.endswith("_top1")])
            if elo_keys:
                print("[val] per-ELO top1 accuracy:")
                for k in elo_keys:
                    bin_name = k.replace('elo_', '').replace('_top1', '')
                    count_key = f"elo_{bin_name}_count"
                    count = val_stats.get(count_key, 0)
                    print(f"  {bin_name}: {val_stats[k]:.3f} (n={count:,})")

            for name, value in train_stats.items():
                writer.add_scalar(f"train_epoch/{name}", value, epoch)
            for name, value in val_stats.items():
                writer.add_scalar(f"val/{name}", value, epoch)

            # End-of-epoch checkpoint (samples_consumed=0 means epoch complete)
            ckpt_path = Path(args.checkpoint_dir) / f"chessformer_{args.run_slug}_epoch{epoch:02d}.pt"
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                ckpt_path, args.config, run_dir, samples_consumed=0,
                config_dict=getattr(args, "config_dict", None),
            )

            val_loss = val_stats.get('loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = Path(args.checkpoint_dir) / f"chessformer_{args.run_slug}_best.pt"
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                    best_path, args.config, run_dir, samples_consumed=0,
                    config_dict=getattr(args, "config_dict", None),
                )

    writer.close()
    print(f"\n[done] Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
