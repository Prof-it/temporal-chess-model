"""
Export Marvin Maia3 PyTorch checkpoints (.pt) to ONNX format.

The exported model uses dynamic batch axes so it works with any batch size,
enabling batched MCTS evaluation for ~4-5x GPU throughput improvement.

Usage:
    python scripts/export_onnx.py --input checkpoints/marvin_23m_cond_1M.pt
    python scripts/export_onnx.py --input checkpoints/marvin_79m_cond.pt --validate --benchmark
    python scripts/export_onnx.py --input checkpoints/marvin_23m_cond_1M.pt \
        -o inference/marvin_23m_cond_1M.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import Maia3Net, NUM_TIME_BINS

INPUT_NAMES = [
    "board_history", "time_history", "rep_flags", "castling",
    "ep_mask", "scalars", "legal_mask",
]
OUTPUT_NAMES = [
    "move_logits", "wdl_logits", "resign_logit", "flag_logit",
    "time_baseline", "time_from_logits", "time_to_logits",
    "promo_logits",
]


def _normalize_state_dict(state_dict: dict) -> dict:
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def _resolve_config(weights_path: Path) -> tuple[dict, str]:
    """Return (config dict, config-name slug) for a Maia3Net checkpoint.

    Checkpoints written by the current `train.py` carry a full `config_dict`
    payload, so the export is fully self-describing. Older checkpoints (or
    bare state_dicts) are rejected with a clear error — train one fresh and
    retry.
    """
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"{weights_path}: expected a training checkpoint dict with a "
            f"'config_dict' key; got a bare state_dict. Retrain with the current "
            f"train.py — checkpoints now persist their full model config."
        )

    saved_dict = ckpt.get("config_dict")
    if not isinstance(saved_dict, dict) or "dim_vit" not in saved_dict:
        raise ValueError(
            f"{weights_path}: missing or unrecognized 'config_dict' (expected a "
            f"Maia3Net config with 'dim_vit'). This .pt was written by an older "
            f"train.py that didn't persist the model architecture. Restart training "
            f"with the current train.py — every checkpoint it writes contains the "
            f"full config dict needed to reconstruct the model."
        )

    config = dict(saved_dict)
    name = ckpt.get("config", "unknown")
    if isinstance(name, str):
        name = name.replace("_", "-").replace("-v2", "")
    print(f"  [config] resolved from checkpoint config_dict (name={name!r})")
    return config, name


class Maia3ONNXWrapper(nn.Module):
    """Wraps Maia3Net to accept flat tensor arguments instead of a dict. ONNX
    doesn't support dict inputs, so we take individual named tensors and repack
    them into the batch dict that ``forward()`` expects.

    Returns all 8 outputs including resign/flag logits and promotion logits.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.num_time_bins = NUM_TIME_BINS

    def forward(
        self,
        board_history: torch.Tensor,   # (B, 8, 64) int64
        time_history: torch.Tensor,    # (B, 8) float32
        rep_flags: torch.Tensor,       # (B, 8) float32
        castling: torch.Tensor,        # (B, 4) float32
        ep_mask: torch.Tensor,         # (B, 64) float32
        scalars: torch.Tensor,         # (B, 8) float32
        legal_mask: torch.Tensor,      # (B, 4096) bool
    ):
        batch = {
            "board_history": board_history,
            "time_history": time_history,
            "rep_flags": rep_flags,
            "castling": castling,
            "ep_mask": ep_mask,
            "scalars": scalars,
            "legal_mask": legal_mask,
        }
        outputs = self.model(batch, return_promo=True)
        (move_logits, wdl_logits, resign_logit, flag_logit,
         time_baseline, time_from_logits, time_to_logits, promo_logits) = outputs

        batch_size = board_history.shape[0]
        device = board_history.device
        dtype = move_logits.dtype

        # The flag head is absent for no_time_context models — emit zeros so the
        # exported ONNX graph keeps a fixed output signature.
        if flag_logit is None:
            flag_logit = torch.zeros(batch_size, dtype=dtype, device=device)

        if time_baseline is None:
            time_baseline = torch.zeros(batch_size, self.num_time_bins, dtype=dtype, device=device)
            time_from_logits = torch.zeros(batch_size, 64, self.num_time_bins, dtype=dtype, device=device)
            time_to_logits = torch.zeros(batch_size, 64, self.num_time_bins, dtype=dtype, device=device)

        return (move_logits, wdl_logits, resign_logit, flag_logit,
                time_baseline, time_from_logits, time_to_logits, promo_logits)


def _create_dummy_inputs(batch_size: int, device: torch.device) -> tuple:
    """Create dummy inputs for ONNX export tracing."""
    return (
        torch.randint(0, 13, (batch_size, 8, 64), dtype=torch.int64, device=device),  # board_history
        torch.randn(batch_size, 8, dtype=torch.float32, device=device),                # time_history
        torch.randint(0, 2, (batch_size, 8), dtype=torch.float32, device=device),      # rep_flags
        torch.randint(0, 2, (batch_size, 4), dtype=torch.float32, device=device),      # castling
        torch.zeros(batch_size, 64, dtype=torch.float32, device=device),               # ep_mask
        torch.randn(batch_size, 9, dtype=torch.float32, device=device),                # scalars (9-wide: see model.py:1318-1321)
        torch.ones(batch_size, 4096, dtype=torch.bool, device=device),                 # legal_mask
    )


class _ManualRMSNorm(nn.Module):
    """ONNX-friendly RMSNorm: y = x * rsqrt(mean(x**2) + eps) * weight.

    nn.RMSNorm maps to aten::rms_norm which isn't supported by the legacy
    ONNX exporter at opset 17 (added in opset 23). This class decomposes the
    same computation into ops that opset 17 understands. Weight is shared
    with the original module (we just wrap it).
    """
    def __init__(self, normalized_shape, eps, weight: nn.Parameter):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        # nn.RMSNorm.eps defaults to None — the C++ impl substitutes
        # torch.finfo(x.dtype).eps (~1.19e-7 for float32). 1e-6 is a safe
        # numerical floor and matches the value used by the Chessformer
        # RMSNorm in model.py.
        self.eps = 1e-6 if eps is None else float(eps)
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        var = x.pow(2).mean(dim=dims, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def _swap_rms_norm_for_onnx(module: nn.Module) -> None:
    """Recursively replace nn.RMSNorm submodules with _ManualRMSNorm so the
    legacy ONNX exporter at opset 17 can trace the graph."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            replacement = _ManualRMSNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                weight=child.weight,
            )
            setattr(module, name, replacement)
        else:
            _swap_rms_norm_for_onnx(child)


def _load_model(weights_path: Path, config: dict, device: torch.device) -> nn.Module:
    """Load a Maia3Net from a .pt checkpoint using the resolved config dict."""
    model = Maia3Net(config)

    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    # Support both full training checkpoints (with 'model' key) and bare state_dicts.
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    # Unwrap DDP / compiled wrappers if present.
    state_dict = _normalize_state_dict(state_dict)

    info = model.load_state_dict(state_dict, strict=False)
    if info.missing_keys:
        print(f"  WARNING: {len(info.missing_keys)} missing keys: {info.missing_keys[:5]}...")
    if info.unexpected_keys:
        print(f"  WARNING: {len(info.unexpected_keys)} unexpected keys: {info.unexpected_keys[:5]}...")

    # Replace nn.RMSNorm with an ONNX-friendly equivalent (Maia3Net uses
    # nn.RMSNorm in its blocks when use_rms_norm=True).
    _swap_rms_norm_for_onnx(model)

    model.to(device)
    model.eval()
    return model


def export_to_onnx(
    weights_path: Path,
    output_path: Path,
    config: dict,
    config_name: str,
    device: str = "cuda",
    opset_version: int = 17,
) -> Path:
    """Export a Maia3Net .pt checkpoint to ONNX with dynamic batch axes."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Loading maia3 ({config_name}) from {weights_path.name} ...")
    print(f"  per_position_clocks={config.get('per_position_clocks')}  "
          f"shrink_global_time={config.get('shrink_global_time')}  "
          f"time_encoder={config.get('time_encoder', 'tokens')}  "
          f"elo_encoder={config.get('elo_encoder', 'endpoints')}")
    model = _load_model(weights_path, config, dev)
    wrapper = Maia3ONNXWrapper(model)
    wrapper.eval()

    # Use batch_size=2 for tracing so the exporter sees a non-degenerate batch dim.
    dummy = _create_dummy_inputs(batch_size=2, device=dev)

    # Dynamic batch axes for every input and output.
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES}

    print(f"Exporting to ONNX (opset {opset_version}, dynamic batch) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,  # Legacy TorchScript exporter for compatibility
        )

    # Validate the graph.
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  OK: {output_path}  ({size_mb:.1f} MB)")
    print(f"  Inputs:  {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    return output_path


def validate_onnx_vs_pytorch(
    weights_path: Path,
    onnx_path: Path,
    config: dict,
    device: str = "cuda",
) -> bool:
    """Compare ONNX outputs against PyTorch outputs for numerical validation."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Validating ONNX vs PyTorch ...")
    print("=" * 60)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = _load_model(weights_path, config, dev)
    model.eval()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"  ONNX Runtime provider: {session.get_providers()[0]}")
    session_input_names = {inp.name for inp in session.get_inputs()}

    all_ok = True
    wrapper = Maia3ONNXWrapper(model).eval()
    for bs in [1, 4, 16]:
        dummy = _create_dummy_inputs(bs, dev)
        with torch.no_grad():
            torch_outs = wrapper(*dummy)

        ort_inputs = {
            n: t.cpu().numpy()
            for n, t in zip(INPUT_NAMES, dummy)
            if n in session_input_names
        }
        onnx_outs = session.run(None, ort_inputs)

        max_diffs = []
        for name, t_out, o_out in zip(OUTPUT_NAMES, torch_outs, onnx_outs):
            diff = float(np.abs(t_out.cpu().numpy() - o_out).max())
            max_diffs.append(diff)
            if diff > 1e-2:
                all_ok = False

        worst = max(max_diffs)
        status = "OK" if worst < 1e-2 else "FAIL"
        print(f"  [{status}] batch={bs:>2d}:  worst max_diff = {worst:.2e}")

    if all_ok:
        print("  All outputs match within tolerance.")
    else:
        print("  WARNING: Some outputs differ -- check model config / weights.")
    return all_ok


def benchmark_inference(onnx_path: Path, device: str = "cuda", num_iters: int = 200) -> None:
    """Benchmark ONNX Runtime inference speed at various batch sizes."""
    import onnxruntime as ort
    import time

    print("\n" + "=" * 60)
    print("Benchmarking ONNX Runtime Inference ...")
    print("=" * 60)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
    print(f"  Provider: {session.get_providers()[0]}")
    session_input_names = {inp.name for inp in session.get_inputs()}

    for bs in [1, 4, 8, 16, 32, 64]:
        dummy = _create_dummy_inputs(bs, dev)
        ort_inputs = {
            n: t.cpu().numpy()
            for n, t in zip(INPUT_NAMES, dummy)
            if n in session_input_names
        }

        # Warmup
        for _ in range(5):
            session.run(None, ort_inputs)

        n = max(20, num_iters // bs)
        t0 = time.perf_counter()
        for _ in range(n):
            session.run(None, ort_inputs)
        elapsed = time.perf_counter() - t0

        evals = n * bs
        ms_call = (elapsed / n) * 1000
        ms_eval = (elapsed / evals) * 1000
        evals_s = evals / elapsed
        print(f"  batch={bs:>2d}: {ms_call:6.2f} ms/call, {ms_eval:5.2f} ms/eval, {evals_s:7.0f} evals/s")


def main():
    parser = argparse.ArgumentParser(
        description="Export Maia3 .pt -> ONNX (dynamic batch axes)",
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Path to a .pt checkpoint (e.g. checkpoints/marvin_23m_cond_1M.pt)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output ONNX path. Default: replaces .pt with .onnx",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda",
        help="Device for export tracing (cuda or cpu)",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument("--validate", action="store_true", help="Validate ONNX vs PyTorch outputs")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")

    args = parser.parse_args()

    weights = args.input.resolve()
    if not weights.exists():
        print(f"Error: {weights} not found")
        sys.exit(1)
    if weights.suffix.lower() != ".pt":
        print(f"Error: expected a .pt checkpoint, got {weights.suffix}")
        sys.exit(1)

    config, config_name = _resolve_config(weights)
    output = args.output or weights.with_suffix(".onnx")

    export_to_onnx(weights, output, config, config_name, device=args.device, opset_version=args.opset)

    if args.validate:
        validate_onnx_vs_pytorch(weights, output, config, device=args.device)

    if args.benchmark:
        benchmark_inference(output, device=args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
