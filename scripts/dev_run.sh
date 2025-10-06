#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export CONFIG_PATH=${CONFIG_PATH:-config.yaml}

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Use the venv python
"$PROJECT_DIR/.venv/bin/python" -m src.main
