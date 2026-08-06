"""Where the inference runtime finds its weights.

Resolution order, first match wins:

1. ``MARVIN_WEIGHTS`` -- a full path to an ``.onnx`` file, anywhere on disk
2. ``MARVIN_MODEL``   -- a filename inside ``inference/``
3. auto-detect        -- whichever ``.onnx`` files are present in ``inference/``

Auto-detect is what makes a fresh checkout work: download any one model with
``python download_assets.py --engine --model <name>`` and the engine picks it up
without further configuration. When several are present, PREFERRED_ORDER decides,
favouring the longest-trained time-conditioned model.
"""

import os
from pathlib import Path

# Most preferred first. Anything not listed still works; it just sorts last.
PREFERRED_ORDER = (
    "marvin_23m_cond_1700k.onnx",   # longest-trained 23M, the usual default
    "marvin_79m_cond_1650k.onnx",   # strongest overall, slower
    "marvin_23m_cond_1M.onnx",
    "marvin_79m_cond_1M.onnx",
    "marvin_23m_blind_1M.onnx",     # time-blind baseline, last resort
)

DEFAULT_CONFIG = "auto"


def get_config_name() -> str:
    """Architecture label. 'auto' resolves to the only architecture there is."""
    return os.environ.get("MARVIN_CONFIG", DEFAULT_CONFIG)


def available_models(repo_root: Path) -> list[Path]:
    """Every ``.onnx`` in ``inference/``, most preferred first."""
    found = sorted((repo_root / "inference").glob("*.onnx"))
    rank = {name: i for i, name in enumerate(PREFERRED_ORDER)}
    return sorted(found, key=lambda p: (rank.get(p.name, len(rank)), p.name))


def get_model_path(repo_root: Path) -> Path:
    """Resolve the weights to load, or raise with instructions if there are none."""
    explicit = os.environ.get("MARVIN_WEIGHTS")
    if explicit:
        return Path(explicit).expanduser()

    named = os.environ.get("MARVIN_MODEL")
    if named:
        return repo_root / "inference" / named

    models = available_models(repo_root)
    if not models:
        raise FileNotFoundError(
            f"No .onnx weights found in {repo_root / 'inference'}.\n"
            "Download one with:\n"
            "    python download_assets.py --engine\n"
            "or point MARVIN_WEIGHTS at an .onnx file.")
    return models[0]


def print_config(repo_root: Path) -> None:
    """Summarise what will load, and what else is available."""
    models = available_models(repo_root)
    if not models:
        print(f"No .onnx weights in {repo_root / 'inference'}. "
              "Run: python download_assets.py --engine")
        return
    chosen = get_model_path(repo_root)
    print(f"Model: {chosen.name} ({'found' if chosen.exists() else 'NOT FOUND'})")
    if len(models) > 1:
        others = ", ".join(p.name for p in models if p != chosen)
        print(f"Also available: {others}")
    print(f"Config: {get_config_name()}")
