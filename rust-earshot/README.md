# Earshot - Rust Conversational AI Client

A high-performance Rust client for real-time voice conversations with AI. Handles audio capture, voice activity detection, and coordinates speech-to-text, language model, and text-to-speech services.

> **Note**: This is part of the [earshot](https://github.com/your-repo/earshot) project. It uses Python servers from the parent directory for STT and TTS.

## Quick Start

```bash
# 1. From the parent earshot directory, start the Python servers (in separate terminals):
cd /path/to/earshot
bash scripts/run_whisper_server.sh   # STT server on port 8765
bash scripts/run_tts_server.sh       # TTS server on port 8766

# 2. Make sure you have an LLM API running (e.g., LM Studio, Ollama)

# 3. Build and run the Rust client:
cd rust-earshot
cargo build --release
./run.sh
```

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     RUST (earshot binary)                      │
│                                                                │
│  [Microphone] ──cpal──> [VAD/Segmentor] ──> [STT Client] ───┐ │
│       │                      │                               │ │
│       │               speech detected                        │ │
│       │                      ▼                               │ │
│       │            ┌─────────────────┐                      │ │
│       │            │ Interrupt Logic │ <───────────────┐    │ │
│       │            └─────────────────┘                 │    │ │
│       │                                                │    │ │
│  [Speaker] <──cpal── [Audio Player] <── [TTS Client] <─┤    │ │
│                           │                            │    │ │
│                           └──── [LLM Client] <─────────┘    │ │
│                                (HTTP/SSE)                    │ │
└────────────────────────────────────────────────────────────────┘
                    │                      │
                    ▼                      ▼
         ┌──────────────────┐   ┌──────────────────┐
         │ PARAKEET SERVER  │   │ CHATTERBOX SERVER│
         │ (Python/NeMo)    │   │ (Python/PyTorch) │
         │ ws://localhost:  │   │ ws://localhost:  │
         │      8765        │   │      8766        │
         └──────────────────┘   └──────────────────┘
```

**Hybrid Design**: Rust handles the coordination, audio I/O, and VAD while Python servers run the ML models (Parakeet for STT, Chatterbox for TTS). This gives you rock-solid async threading without Python's GIL issues.

## Features

- **Real-time VAD**: Silero VAD via ONNX Runtime with configurable thresholds
- **Speech Segmentation**: Pre/post padding, hangover detection, minimum speech duration
- **Auto-reconnect**: Automatically reconnects to STT/TTS servers on connection drops
- **Interrupt Handling**: Detects speech during playback, cancels generation, fades out audio
- **Pipeline State Tracking**: Accurate state machine that waits for full pipeline completion
- **Streaming LLM**: Server-sent events (SSE) streaming with sentence-level chunking
- **Notification System**: HTTP endpoint for push notifications that are read aloud

## Prerequisites

### 1. ONNX Runtime (for VAD inference)

The Rust client needs ONNX Runtime to run the Silero VAD model. Options:

```bash
# Option A: Use ONNX Runtime from the Python venv (if earshot venv is set up)
# Edit run.sh to point to:
export ORT_DYLIB_PATH="/path/to/earshot/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.0"

# Option B: Download ONNX Runtime directly
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz
# Then set ORT_DYLIB_PATH to ./onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
```

### 2. Silero VAD Model

The VAD model should be at `../models/silero_vad.onnx` (relative to rust-earshot). This is shared with the parent earshot project.

Download if needed:
```bash
wget -P ../models/ https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

### 3. Python Servers (from parent earshot project)

Start these from the **parent earshot directory**:

```bash
# Terminal 1 - STT Server (Parakeet/Whisper on port 8765)
bash scripts/run_whisper_server.sh

# Terminal 2 - TTS Server (Chatterbox on port 8766)
bash scripts/run_tts_server.sh
```

### 4. LLM API (OpenAI-compatible)

Any OpenAI-compatible API:
- [LM Studio](https://lmstudio.ai/) - Local GUI
- [Ollama](https://ollama.ai/) - `ollama serve`
- [vLLM](https://github.com/vllm-project/vllm) - Production server
- OpenAI API - Cloud

Update `config.yaml` with your LLM endpoint.

## Building

```bash
# Development build
cargo build

# Release build (recommended)
cargo build --release

# Cross-compile for Windows (from Linux)
cargo build --release --target x86_64-pc-windows-gnu --no-default-features --features rustls
```

## Running

```bash
# Use the run script (recommended)
./run.sh

# Or set ONNX Runtime path manually
ORT_DYLIB_PATH=/path/to/libonnxruntime.so ./target/release/earshot
```

## Configuration

Edit `config.yaml`:

```yaml
audio:
  sample_rate: 16000      # Audio sample rate (Hz)
  channels: 1             # Mono audio
  frame_ms: 30            # Frame size in milliseconds

vad:
  model_path: "../models/silero_vad.onnx"  # Path to VAD model
  start_threshold: 0.35   # Probability to start speech detection
  end_threshold: 0.25     # Probability to end speech detection
  ema_alpha: 0.30         # EMA smoothing factor
  pre_ms: 600             # Pre-speech padding (ms)
  hang_ms: 700            # Silence tolerance before ending (ms)
  post_ms: 400            # Post-speech padding (ms)
  max_segment_s: 30.0     # Maximum segment length (seconds)
  min_start_frames: 3     # Consecutive voiced frames to start
  min_speech_ms: 700      # Minimum speech duration (ms)

stt:
  url: "ws://localhost:8765"  # STT server URL

tts:
  url: "ws://localhost:8766"  # TTS server URL

llm:
  url: "http://localhost:1234/v1/chat/completions"  # LLM API endpoint
  model: "your-model-name"
  temperature: 0.7
  system_prompt: "You are a helpful voice assistant..."

notifications:
  enabled: true
  port: 9999              # HTTP endpoint for notifications
  idle_poll_threshold_s: 10
  idle_immediate_threshold_s: 5
```

## Notifications

Send notifications via HTTP that will be read aloud:

```bash
curl -X POST http://localhost:9999/notify \
  -H "Content-Type: application/json" \
  -d '{"title": "Reminder", "body": "Meeting in 5 minutes", "source": "calendar"}'
```

Notifications are batched and read when the system is idle (no active conversation).

## Project Structure

```
rust-earshot/
├── Cargo.toml
├── config.yaml
├── run.sh
└── src/
    ├── main.rs              # Entry point
    ├── config.rs            # YAML config loading
    ├── logging.rs           # Conversation logger
    ├── audio/
    │   ├── capture.rs       # Mic input (cpal)
    │   └── playback.rs      # Speaker output with fade-out
    ├── vad/
    │   ├── silero.rs        # ONNX VAD inference
    │   └── segmentor.rs     # Speech segment detection
    ├── clients/
    │   ├── stt_client.rs    # WebSocket to STT server
    │   ├── tts_client.rs    # WebSocket to TTS server
    │   └── llm_client.rs    # HTTP/SSE streaming
    ├── notifications/
    │   ├── server.rs        # HTTP notification endpoint
    │   ├── queue.rs         # Notification batching
    │   └── types.rs         # Notification types
    └── pipeline/
        └── conversation.rs  # State machine + coordination
```

## State Machine

The client operates in two states:

- **IDLE**: Listening for speech, VAD running
- **PROCESSING**: STT → LLM → TTS pipeline active

Transitions:
- IDLE → PROCESSING: Speech segment detected
- PROCESSING → IDLE: All audio finished playing
- PROCESSING → PROCESSING (interrupt): New speech detected during playback

## Troubleshooting

**"cannot open shared object file: libonnxruntime.so"**
- Set `ORT_DYLIB_PATH` to point to the ONNX Runtime library
- Check that the path in `run.sh` is correct

**"Connection refused" on port 8765 or 8766**
- Start the Python servers from the parent directory:
  ```bash
  cd ..
  bash scripts/run_whisper_server.sh
  bash scripts/run_tts_server.sh
  ```

**"Broken pipe" errors**
- The WebSocket connection timed out (normal after ~60s idle)
- Auto-reconnect will handle this automatically

**State stuck in PROCESSING**
- Check that all servers (STT, TTS, LLM) are running
- Check the logs for connection errors

**No audio output**
- Check your system's default audio output device
- Verify the TTS server is returning audio (check its logs)
