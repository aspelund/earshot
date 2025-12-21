#!/bin/bash
# Start the v2 conversational client with robust cancellation
#
# Prerequisites:
# 1. Start Whisper server: bash scripts/run_whisper_server.sh
# 2. Start TTS server: bash scripts/run_tts_server.sh
# 3. Start LLM server (e.g., LM Studio on Windows host)

cd "$(dirname "$0")/.."

echo "Starting conversational client v2..."
echo "Using config: config.conversational_client_v2.yaml"
echo ""
echo "Make sure these servers are running:"
echo "  - Whisper server (ws://localhost:8765)"
echo "  - TTS server (ws://localhost:8766)"
echo "  - LLM server (OpenAI-compatible API)"
echo ""

.venv/bin/python scripts/conversational_client_v2.py --config config.conversational_client_v2.yaml
