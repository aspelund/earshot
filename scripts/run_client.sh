#!/bin/bash
# Run the audio streaming client (capture mic and send to server)

SERVER_URL="${1:-ws://localhost:8765}"

echo "Streaming microphone audio to $SERVER_URL"
echo "Usage: $0 [server_url]"
echo "Example: $0 ws://192.168.1.100:8765"
echo ""

# Activate venv if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run streaming client
python scripts/stream_client.py --server "$SERVER_URL"
