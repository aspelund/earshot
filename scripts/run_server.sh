#!/bin/bash
# Run the STT pipeline in server mode (listening for network audio stream)

# Update config to use network source
export CONFIG_PATH="${CONFIG_PATH:-config.yaml}"

echo "Starting STT server in network mode..."
echo "Edit config.yaml to set audio.source='network' and configure network settings"
echo ""

# Activate venv if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run main pipeline
python -m src.main
