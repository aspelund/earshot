#!/usr/bin/env bash
set -euo pipefail

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default config and server
CONFIG="${1:-$PROJECT_DIR/config.vad_client.yaml}"
SERVER="${2:-ws://localhost:8765}"
LANGUAGE="${3:-}"

echo "VAD Client Configuration:"
echo "  Config: $CONFIG"
echo "  Server: $SERVER"
if [ -n "$LANGUAGE" ]; then
  echo "  Language: $LANGUAGE"
fi
echo ""

# Build command
CMD="$PROJECT_DIR/.venv/bin/python $SCRIPT_DIR/vad_client.py --config $CONFIG --server $SERVER"

# Add language if specified
if [ -n "$LANGUAGE" ]; then
  CMD="$CMD --language $LANGUAGE"
fi

# Execute
eval "$CMD"
