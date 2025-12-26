# Earshot

Local, open-weight, real-time speech-to-speech AI assistant. Fully offline conversational AI using state-of-the-art open models.

## Overview

Earshot is a complete voice assistant stack:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RUST CLIENT (rust-earshot)                    │
│  Microphone → VAD → STT Client → LLM Client → TTS Client → Speaker  │
│                        + GUI + Notifications                         │
└─────────────────────────────────────────────────────────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│   STT Server      │  │   TTS Server      │  │   LLM Server      │
│ (Parakeet/Whisper)│  │ (Soprano/Chatter) │  │ (LM Studio/Ollama)│
│   Python/NeMo     │  │   Python/PyTorch  │  │   Your choice     │
│   Port 8765       │  │   Port 8766       │  │   Port 1234       │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

**Components:**
- **Rust Client** (`rust-earshot/`): High-performance audio I/O, VAD, and pipeline coordination with GUI
- **Python STT Server**: Parakeet (NVIDIA NeMo) or faster-whisper for speech-to-text
- **Python TTS Server**: Soprano (streaming, 64ms latency) or Chatterbox for text-to-speech
- **LLM**: Any OpenAI-compatible API (LM Studio, Ollama, vLLM, etc.)

## Features

- **Real-time VAD**: Silero VAD with configurable thresholds
- **Streaming TTS**: ~64ms time-to-first-audio with Soprano
- **Interrupt handling**: Speak to interrupt AI mid-response
- **GUI**: Desktop application with audio level meters
- **Notifications**: HTTP endpoint for push notifications read aloud
- **Cross-platform**: Linux and Windows support
- **Fully local**: No cloud dependencies, runs on consumer hardware

## Quick Start

```bash
# 1. Clone and setup Python environment
git clone <repo-url> earshot
cd earshot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download models
python scripts/install_models.py

# 3. Start servers (in separate terminals)
bash scripts/run_whisper_server.sh   # STT on port 8765
bash scripts/run_tts_server.sh       # TTS on port 8766

# 4. Start an LLM server (e.g., LM Studio on port 1234)

# 5. Run the Rust client
cd rust-earshot
./run.sh
```

See [`rust-earshot/README.md`](rust-earshot/README.md) for detailed Rust client documentation.

---

# Python Servers

## STT Server Features

- **Real-time VAD**: Silero VAD for accurate speech detection with hysteresis
- **Async STT**: faster-whisper on worker thread (never blocks audio)
- **Multilingual**: Supports 99 languages with auto-detection
- **Smart segmentation**: Pre/post padding, hangover, min duration filtering
- **JSONL logging**: Structured logs with UTC timestamps + word-level timing
- **Network streaming**: Stream audio from remote devices (laptop/phone to server)

## TTS Server Features

- **Soprano backend**: Streaming output with ~64ms latency per chunk
- **Chatterbox backend**: High-quality batch synthesis
- **Emoji filtering**: Automatic removal of unspeakable characters
- **WebSocket API**: Binary audio streaming

## Setup on New Machine

### Prerequisites

**For local microphone mode:**
- macOS (Apple Silicon or Intel)
- Python 3.9+ (`python3 --version`)
- Homebrew (optional, for system dependencies)

**For network streaming mode (server without mic):**
- Linux or macOS
- Python 3.9+
- No microphone required (receives audio over network)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url> vad-server
cd vad-server

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies (~2 min)
pip install -r requirements.txt

# 4. Download VAD model
python scripts/install_models.py

# 5. Download STT model (choose one)
# For multilingual (medium): ~1.5GB, best accuracy
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"

# For English only (base.en): ~145MB, faster
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

# 6. Configure (optional)
cp .env.example .env
# Edit config.yaml to change model, thresholds, etc.

# 7. Grant microphone access
# Run once to trigger permission prompt:
python scripts/test_mic.py

# 8. Test the pipeline
bash scripts/dev_run.sh
# Speak into your mic - transcriptions appear in terminal

# 9. In another terminal, watch logs
tail -f ~/stt_logs/asr.jsonl
```

### Quick Test Scripts

```bash
# Test microphone capture (visual level meter)
python scripts/test_mic.py

# Test VAD detection (shows speech probability)
python scripts/test_vad.py

# Test STT directly (record then transcribe)
python scripts/test_stt.py
```

## Network Streaming Mode

Run the STT pipeline on a server without a microphone by streaming audio from a remote client.

### Server Setup (Linux/macOS without mic)

```bash
# 1. Use the server config (network audio source)
cp config.server.yaml config.yaml

# OR manually edit config.yaml:
# Set audio.source: "network"

# 2. Start the server
source .venv/bin/activate
python -m src.main

# Server will listen on ws://0.0.0.0:8765
```

### Client Setup (device with microphone)

On your laptop, phone, or any device with a microphone:

```bash
# 1. Install dependencies (only need sounddevice and websockets)
pip install sounddevice websockets numpy

# 2. Stream audio to server
python scripts/stream_client.py --server ws://SERVER_IP:8765

# Example: Stream to server at 192.168.1.100
python scripts/stream_client.py --server ws://192.168.1.100:8765
```

**The client will:**
- Stream your microphone audio to the server
- Receive and display transcriptions in real-time
- Show transcription latency for each segment

### Configuration

**Server (`config.yaml` or `config.server.yaml`):**
```yaml
audio:
  source: "network"  # Use network stream instead of microphone

network:
  host: "0.0.0.0"    # Listen on all interfaces
  port: 8765         # WebSocket port
  auth_token: null   # Optional: set to "your-secret-token" for auth
```

**Client:**
```bash
# With authentication
python scripts/stream_client.py \
  --server ws://192.168.1.100:8765 \
  --auth-token your-secret-token
```

## Configuration

Edit `config.yaml` to adjust:
- Audio source (mic vs network)
- VAD thresholds and timing
- STT model size (tiny.en, base.en, small.en)
- Log rotation settings
- Heartbeat interval
- Network server settings (host, port, auth)

## Testing

See `TESTING.md` for comprehensive testing procedures.

Run automated tests:
```bash
pytest tests/
```

## Installation as Daemon

Run at login automatically:

```bash
# 1. Update paths in plist if needed
# Edit launchd/com.local.stt.daemon.plist to match your install location

# 2. Copy plist to LaunchAgents
cp launchd/com.local.stt.daemon.plist ~/Library/LaunchAgents/

# 3. Load and start
launchctl load ~/Library/LaunchAgents/com.local.stt.daemon.plist
launchctl start com.local.stt.daemon

# 4. Check status
launchctl list | grep stt.daemon

# 5. View logs
tail -f /tmp/stt-daemon.err
tail -f ~/stt_logs/asr.jsonl

# To stop/unload
launchctl stop com.local.stt.daemon
launchctl unload ~/Library/LaunchAgents/com.local.stt.daemon.plist
```

## Log Format

Each line is a JSON object:
```json
{
  "type": "asr_segment",
  "start_utc": "2025-10-06T07:59:12.184Z",
  "end_utc": "2025-10-06T07:59:13.902Z",
  "latency_s": 0.17,
  "text": "okay let's kick off",
  "tokens": [
    {"w": "okay", "start_ms": 1696582752184, "end_ms": 1696582752500}
  ]
}
```

## Performance Tuning

### VAD Settings (in `config.yaml`)
- **More sensitive:** Lower `threshold` to 0.35-0.40 (catches quieter speech)
- **Less sensitive:** Raise `threshold` to 0.50-0.55 (fewer false positives)
- **Longer utterances:** Increase `hang_ms` to 800-1000 (holds through pauses)
- **Shorter segments:** Decrease `hang_ms` to 400-500 (cuts faster)
- **Smoother detection:** Lower `ema_alpha` to 0.20-0.25 (less reactive to spikes)

### STT Model Selection
- **tiny.en** - Fastest, lowest accuracy (~75MB)
- **base.en** - Good balance (~145MB)
- **small.en** - Better accuracy (~466MB)
- **medium** - Best multilingual (~1.5GB) ← Current default
- **distil-medium.en** - 2x faster than medium, similar accuracy

Change `model_size` in `config.yaml` and restart.

## Troubleshooting

### Network Streaming Issues

**Server not receiving audio:**
- Check firewall allows port 8765 (or your configured port)
- Verify server IP address: `ip addr` (Linux) or `ifconfig` (macOS)
- Ensure both devices on same network (or port forwarding configured)
- Check server logs for connection messages

**Client connection refused:**
- Verify server is running and listening on correct port
- Test with: `nc -zv SERVER_IP 8765` or `telnet SERVER_IP 8765`
- Check firewall on server allows incoming connections

**Authentication failed:**
- Ensure auth_token matches in both server config and client command
- Token is case-sensitive

### No audio detected
- Check microphone permissions: System Settings → Privacy & Security → Microphone
- Run `python scripts/test_mic.py` to verify audio capture
- Try lowering `vad.threshold` to 0.35

### Transcriptions cut off mid-sentence
- Increase `hang_ms` to 800-1000 in `config.yaml`
- Check `min_speech_ms` isn't too high (try 500)

### High CPU usage
- Use smaller model: `tiny.en` or `base.en`
- Check if model is quantized: `compute_type: "int8"`

### Model download fails
- Manually download: `python -c "from faster_whisper import WhisperModel; WhisperModel('medium')"`
- Models cache to `~/.cache/huggingface/`

### Daemon won't start
- Check `/tmp/stt-daemon.err` for errors
- Verify paths in `launchd/com.local.stt.daemon.plist`
- Ensure venv exists: `ls .venv/bin/python`

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.9+
- ~2GB disk space (for models)
- Microphone access
