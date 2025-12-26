#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export CONFIG_PATH=${CONFIG_PATH:-config.yaml}

# Disable CUDA graphs for NeMo (required for WSL2 compatibility)
export CUDA_LAUNCH_BLOCKING=1
export NEMO_DISABLE_CUDA_GRAPH_DECODER=1

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Add cuDNN libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$PROJECT_DIR/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# Use the venv python
"$PROJECT_DIR/.venv/bin/python" -m src.main_whisper_server
