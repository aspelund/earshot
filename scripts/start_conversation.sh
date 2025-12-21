#!/bin/bash
# Convenience script to start conversational voice assistant

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Conversational Voice Assistant${NC}"
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if Whisper server is running
echo -e "${YELLOW}Checking Whisper server...${NC}"
if ! nc -z localhost 8765 2>/dev/null; then
    echo -e "${RED}Error: Whisper server not running on localhost:8765${NC}"
    echo "Start it in another terminal:"
    echo "  bash scripts/run_whisper_server.sh"
    echo
    echo "Note: Use run_whisper_server.sh (not python -m directly) to set CUDA paths"
    exit 1
fi
echo -e "${GREEN}✓ Whisper server is running${NC}"

# Check if LLM server is running
echo -e "${YELLOW}Checking LLM server...${NC}"
if ! nc -z localhost 1234 2>/dev/null; then
    echo -e "${RED}Warning: LLM server not running on localhost:1234${NC}"
    echo "Start your llama.cpp server or update config with correct host"
    echo
fi

# Check if TTS server is running
echo -e "${YELLOW}Checking TTS server...${NC}"
if ! nc -z localhost 8766 2>/dev/null; then
    echo -e "${RED}Error: TTS server not running on localhost:8766${NC}"
    echo "Start it in another terminal:"
    echo "  bash scripts/run_tts_server.sh"
    echo
    exit 1
fi
echo -e "${GREEN}✓ TTS server is running${NC}"

# Check PulseAudio for audio playback (WSL2)
if grep -qi microsoft /proc/version; then
    echo -e "${YELLOW}Detected WSL2, checking PulseAudio...${NC}"
    if ! pgrep -x pulseaudio > /dev/null; then
        echo -e "${YELLOW}Starting PulseAudio...${NC}"
        pulseaudio --start 2>/dev/null || echo -e "${RED}Warning: Could not start PulseAudio${NC}"
    else
        echo -e "${GREEN}✓ PulseAudio is running${NC}"
    fi
fi

echo
echo -e "${GREEN}Starting conversational client...${NC}"
echo -e "${YELLOW}Speak to chat with the AI assistant. Press Ctrl+C to stop.${NC}"
echo

# Start the client
python scripts/conversational_client.py "$@"
