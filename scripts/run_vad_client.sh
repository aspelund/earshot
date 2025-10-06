#!/usr/bin/env bash
set -euo pipefail

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default config and server
CONFIG="${1:-$PROJECT_DIR/config.vad_client.yaml}"
SERVER="${2:-ws://localhost:8765}"

echo "VAD Client Configuration:"
echo "  Config: $CONFIG"
echo "  Server: $SERVER"
echo ""

# Use the venv python
"$PROJECT_DIR/.venv/bin/python" "$SCRIPT_DIR/vad_client.py" \
  --config "$CONFIG" \
  --server "$SERVER"
