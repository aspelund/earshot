# Earshot - Rust Conversational AI Client

A high-performance Rust client for real-time voice conversations with AI. Handles audio capture, voice activity detection, and coordinates speech-to-text, language model, and text-to-speech services.

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

## Prerequisites

1. **ONNX Runtime** (for VAD inference)
   - Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
   - Extract to project directory or set `ORT_DYLIB_PATH`

2. **Python Servers** (from parent earshot project)
   - Parakeet STT server: `python -m src.main_whisper_server`
   - Chatterbox TTS server: `python -m src.main_tts_server`

3. **LLM API** (OpenAI-compatible)
   - LM Studio, Ollama, or any OpenAI-compatible API

4. **Silero VAD Model**
   - Place `silero_vad.onnx` in the `models/` directory

## Setup

```bash
# Build
cargo build --release

# Set up ONNX Runtime (if not in standard path)
# Option 1: Symlink the library folder
ln -s ../rust-example/onnxruntime-linux-x64-1.22.0 .

# Option 2: Set environment variable when running
export ORT_DYLIB_PATH=./onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
```

## Running

```bash
# Option 1: Use the run script (sets ORT_DYLIB_PATH automatically)
./run.sh

# Option 2: Set environment variable manually
ORT_DYLIB_PATH=./onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so ./target/release/earshot

# Option 3: During development
ORT_DYLIB_PATH=./onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so cargo run --release
```

## Configuration

Edit `config.yaml` to configure the client:

```yaml
audio:
  sample_rate: 16000      # Audio sample rate (Hz)
  channels: 1             # Mono audio
  frame_ms: 30            # Frame size in milliseconds

vad:
  model_path: "models/silero_vad.onnx"  # Path to VAD model
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
  url: "ws://localhost:8765"  # Parakeet server URL

tts:
  url: "ws://localhost:8766"  # Chatterbox server URL

llm:
  url: "http://localhost:1234/v1/chat/completions"  # LLM API endpoint
  model: "qwen/qwen3-4b-2507"  # Model name
  temperature: 0.7
  system_prompt: "You are a helpful voice assistant..."
```

## Project Structure

```
rust-earshot/
├── Cargo.toml
├── config.yaml
├── run.sh
├── models/
│   └── silero_vad.onnx
└── src/
    ├── main.rs              # Entry point
    ├── config.rs            # YAML config loading
    ├── audio/
    │   ├── capture.rs       # Mic input (cpal)
    │   └── playback.rs      # Speaker output with fade-out
    ├── vad/
    │   ├── silero.rs        # ONNX VAD inference
    │   └── segmentor.rs     # Speech segment detection
    ├── clients/
    │   ├── stt_client.rs    # WebSocket to Parakeet
    │   ├── tts_client.rs    # WebSocket to Chatterbox
    │   └── llm_client.rs    # HTTP/SSE streaming
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
- Or symlink the onnxruntime folder to the project directory

**"Broken pipe" errors**
- The WebSocket connection timed out (normal after ~60s idle)
- Auto-reconnect will handle this automatically

**State stuck in PROCESSING**
- Check that all servers (STT, TTS, LLM) are running
- Check the logs for connection errors
