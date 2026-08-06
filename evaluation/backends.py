"""Inference backends for evaluation.

Two backends share one calling convention so evaluation code is agnostic to
the underlying runtime:

    backend(batch_dict, return_promo=True) -> tuple of 8 numpy arrays
        (move_logits, wdl_logits, resign_logit, flag_logit, time_baseline,
         time_from_logits, time_to_logits, promo_logits)

``OnnxBackend`` (re-exported from ``inference.backend``) runs an exported
``.onnx`` graph and is what deployment uses. ``TorchBackend`` loads a training
``.pt`` checkpoint and rebuilds the model from the ``config_dict`` that
``train.py`` persists, so *any* ablation variant is evaluable with no export
step. ``make_backend`` dispatches on file extension.

For ``no_time_context`` checkpoints the flag and time heads are absent; like the
ONNX export wrapper, ``TorchBackend`` emits zero tensors of the correct shape so
the 8-tuple signature is fixed regardless of config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inference.backend import OnnxBackend

# Input dtypes the model's forward() expects (see scripts/export_onnx.py and
# model.Maia3Net.forward). board_history is indices; legal_mask gates the policy.
_INPUT_DTYPES = {
    "board_history": "int64",
    "time_history": "float32",
    "rep_flags": "float32",
    "castling": "float32",
    "ep_mask": "float32",
    "scalars": "float32",
    "legal_mask": "bool",
}

_OUTPUT_NAMES = (
    "move_logits", "wdl_logits", "resign_logit", "flag_logit",
    "time_baseline", "time_from_logits", "time_to_logits", "promo_logits",
)


def _normalize_state_dict(state_dict: dict) -> dict:
    """Strip DDP / torch.compile wrapper prefixes from state-dict keys."""
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


class TorchBackend:
    """Evaluate a Maia3Net ``.pt`` checkpoint directly in PyTorch.

    Mirrors :class:`inference.backend.OnnxBackend`: callable on a dict of numpy
    arrays, returning a fixed 8-tuple of numpy arrays.
    """

    def __init__(self, weights_path: str | Path, device: str = "cuda", config_name: str = "",
                 config: dict | None = None) -> None:
        import torch
        from model import Maia3Net, NUM_TIME_BINS

        self._torch = torch
        self._num_time_bins = NUM_TIME_BINS

        dev = str(device).split(":")[0] if device else "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            print("[TorchBackend] CUDA requested but unavailable — using CPU")
            dev = "cpu"
        self._device = torch.device(dev)

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise ValueError(f"{weights_path}: not a training checkpoint (no 'model' key).")

        # Prefer the checkpoint's self-describing config_dict; fall back to an
        # explicit override for legacy checkpoints (pre-config_dict) where the
        # caller knows the architecture. No silent name-based guessing.
        if config is not None:
            self._config = dict(config)
        elif isinstance(ckpt.get("config_dict"), dict):
            self._config = dict(ckpt["config_dict"])
        else:
            raise ValueError(
                f"{weights_path}: no 'config_dict' in checkpoint and no explicit "
                f"config override given (checkpoint config name={ckpt.get('config')!r}). "
                f"This is a pre-config_dict checkpoint — retrain with the current "
                f"train.py, or pass config=<dict> to TorchBackend."
            )
        self._config_name = config_name or str(ckpt.get("config", "unknown")).replace("_", "-")
        self._no_time_context = bool(self._config.get("no_time_context", False))

        model = Maia3Net(self._config)
        state_dict = _normalize_state_dict(ckpt["model"])
        info = model.load_state_dict(state_dict, strict=False)
        if info.missing_keys:
            print(f"[TorchBackend] {len(info.missing_keys)} missing keys: {info.missing_keys[:5]}")
        if info.unexpected_keys:
            print(f"[TorchBackend] {len(info.unexpected_keys)} unexpected keys: {info.unexpected_keys[:5]}")

        model.to(self._device).eval()
        self._model = model
        print(f"[TorchBackend] Loaded {self._config_name} on {self._device} "
              f"(no_time_context={self._no_time_context})")

    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        return self._device.type

    @property
    def config_name(self) -> str:
        return self._config_name

    @property
    def config(self) -> dict:
        return self._config

    @property
    def has_time_head(self) -> bool:
        return not self._no_time_context

    @property
    def kind(self) -> str:
        return "torch"

    # ------------------------------------------------------------------
    def _to_tensor(self, name: str, arr: np.ndarray):
        torch = self._torch
        np_dtype = {
            "int64": np.int64, "float32": np.float32, "bool": np.bool_,
        }[_INPUT_DTYPES[name]]
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np_dtype)).to(self._device)

    def __call__(self, batch: dict[str, np.ndarray], return_promo: bool = True) -> tuple:
        torch = self._torch
        feed = {name: self._to_tensor(name, batch[name])
                for name in _INPUT_DTYPES if name in batch}
        batch_size = feed["board_history"].shape[0]

        with torch.no_grad():
            outputs = self._model(feed, return_promo=True)
        (move_logits, wdl_logits, resign_logit, flag_logit,
         time_baseline, time_from_logits, time_to_logits, promo_logits) = outputs

        # Backfill heads that are absent for no_time_context models so the
        # 8-tuple shape is identical to the ONNX export.
        T = self._num_time_bins
        dtype, device = move_logits.dtype, move_logits.device
        if flag_logit is None:
            flag_logit = torch.zeros(batch_size, dtype=dtype, device=device)
        if time_baseline is None:
            time_baseline = torch.zeros(batch_size, T, dtype=dtype, device=device)
            time_from_logits = torch.zeros(batch_size, 64, T, dtype=dtype, device=device)
            time_to_logits = torch.zeros(batch_size, 64, T, dtype=dtype, device=device)

        result = (move_logits, wdl_logits, resign_logit, flag_logit,
                  time_baseline, time_from_logits, time_to_logits, promo_logits)
        return tuple(t.detach().cpu().numpy() for t in result)


def make_backend(weights_path: str | Path, device: str = "cuda", config_name: str = "",
                 config: dict | None = None):
    """Return the appropriate backend for a checkpoint, dispatching on suffix.

    ``.pt`` -> :class:`TorchBackend`, ``.onnx`` -> :class:`OnnxBackend`. The
    optional ``config`` override is only honoured by the PyTorch backend.
    """
    suffix = Path(weights_path).suffix.lower()
    if suffix == ".pt":
        return TorchBackend(weights_path, device=device, config_name=config_name, config=config)
    if suffix == ".onnx":
        return OnnxBackend(weights_path, device=device, config_name=config_name)
    raise ValueError(f"Unsupported checkpoint extension {suffix!r} for {weights_path}")
