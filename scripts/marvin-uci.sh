#!/usr/bin/env bash
# Marvin UCI engine launcher for Linux/macOS chess GUIs.
#
# Point Cute Chess / Arena / BanksiaGUI at this file as the engine. It resolves
# the repository from its own location, so the repo can live anywhere as long as
# this file stays in scripts/.
#
# One-time setup, from the repository root:
#     python -m venv venv
#     venv/bin/pip install -r requirements-engine.txt
#     venv/bin/python download_assets.py --engine
#
# Whichever model you downloaded is picked up automatically. Optional overrides:
#     MARVIN_WEIGHTS   path to a specific .onnx file
#     MARVIN_DEVICE    cuda or cpu (default: cuda, falls back to cpu)

set -eu

cd "$(dirname "$0")/.."

export MARVIN_DEVICE="${MARVIN_DEVICE:-cuda}"

PYTHON="venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

# -u keeps stdio unbuffered, which UCI streaming requires. Weights are resolved
# by inference/config.py, so none are passed unless MARVIN_WEIGHTS is set.
if [ -n "${MARVIN_WEIGHTS:-}" ]; then
    exec "$PYTHON" -u -m inference.uci_engine --weights "$MARVIN_WEIGHTS" "$@"
fi
exec "$PYTHON" -u -m inference.uci_engine "$@"
