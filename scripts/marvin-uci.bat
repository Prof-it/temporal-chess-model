@echo off
REM ===========================================================================
REM  Marvin UCI engine launcher for Windows chess GUIs.
REM
REM  Point Arena / Cute Chess / BanksiaGUI / Nibbler at THIS FILE as the engine.
REM  It resolves the repository from its own location, so the repo can live
REM  anywhere as long as this file stays in scripts/.
REM
REM  One-time setup, from the repository root:
REM      python -m venv venv
REM      venv\Scripts\pip install -r requirements-engine.txt
REM      venv\Scripts\python download_assets.py --engine
REM
REM  Whichever model you downloaded is picked up automatically.
REM  Optional overrides:
REM      MARVIN_WEIGHTS   path to a specific .onnx file
REM      MARVIN_DEVICE    cuda or cpu  (default: cuda, falls back to cpu)
REM
REM  GPU needs onnxruntime-gpu plus a matching CUDA / cuDNN 9.x runtime;
REM  without it the engine runs on CPU automatically.
REM ===========================================================================

setlocal

REM %~dp0 is this file's directory (scripts\); .. is the repository root.
pushd "%~dp0.."

if not defined MARVIN_DEVICE set "MARVIN_DEVICE=cuda"

REM Prefer the project venv, else fall back to python on PATH.
set "PYTHON=venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

REM -u keeps stdio unbuffered, which UCI streaming requires. Weights are
REM resolved by inference\config.py unless MARVIN_WEIGHTS names one.
if defined MARVIN_WEIGHTS (
    "%PYTHON%" -u -m inference.uci_engine --weights "%MARVIN_WEIGHTS%" %*
) else (
    "%PYTHON%" -u -m inference.uci_engine %*
)

popd
