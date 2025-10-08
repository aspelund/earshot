# Conversational Voice Assistant

An interactive voice assistant that combines VAD, Whisper transcription, LLM (Qwen), and TTS for natural conversations.

## Architecture

```
[Microphone] → [VAD Client] → [Whisper Server] → [Transcription]
                    ↓                                    ↓
              [LLM (Qwen)] ← ← ← ← ← ← ← ← ← ← ← ← ← ←
                    ↓
              [TTS Server]
                    ↓
              [Audio Playback]
```

## Setup

### 1. Start the Whisper Server

```bash
bash scripts/run_whisper_server.sh
```

**Important**: Use the `run_whisper_server.sh` script (not `python -m` directly) as it sets up the required `LD_LIBRARY_PATH` for CUDA/cuDNN libraries.

### 2. Start LLM Server (llama.cpp)

Make sure your Qwen model is running on `localhost:1234`:

```bash
# Example using llama.cpp server
./llama-server -m models/qwen3-4b-2507.gguf -c 4096 --port 1234
```

### 3. Start TTS Server

Make sure your TTS server is running on `localhost:3500`:

```bash
# Example TTS server command
your-tts-server --port 3500
```

### 4. Start Conversational Client

```bash
source .venv/bin/activate
python scripts/conversational_client.py
```

## Configuration

Edit `config.conversational_client.yaml`:

### WSL2 Networking

If `localhost` doesn't work in WSL2, you have several options:

**Option 1: Use Windows host IP**
```yaml
llm:
  host: "172.x.x.x"  # Your Windows host IP
tts:
  host: "172.x.x.x"
```

Find Windows host IP from WSL2:
```bash
# Method 1: From /etc/resolv.conf
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'

# Method 2: Using hostname
hostname -I | awk '{print $1}'
```

**Option 2: Use host.docker.internal (if available)**
```yaml
llm:
  host: "host.docker.internal"
tts:
  host: "host.docker.internal"
```

### Customize System Prompt

```yaml
llm:
  system_prompt: "You are a helpful voice assistant. Keep responses concise and natural for speech."
```

### Adjust Audio Settings

```yaml
playback:
  fade_out_duration_ms: 250  # Fade-out speed when interrupted

conversation:
  interrupt_timeout_s: 0.5  # Wait time after interrupt before processing
```

## How It Works

### Normal Flow

1. **User speaks** → VAD detects speech
2. **Segment sent** to Whisper server
3. **Transcription received**
4. **LLM generates** response (with conversation history)
5. **TTS synthesizes** audio
6. **Audio plays back** to user

### Interruption Handling

#### Before Audio Playback Starts

If you speak while LLM/TTS is processing:
- **Cancels** pending LLM/TTS calls
- **Accumulates** your messages
- **Waits** 0.5s for you to finish
- **Combines** all messages and retries

Example:
- You: "What's the weather?" → LLM processing...
- You: "Actually, make that for tomorrow" → Cancels
- Combined message sent: "What's the weather? Actually, make that for tomorrow"

#### After Audio Playback Starts

If you speak while AI is talking:
- **Fades out** audio (250ms)
- **Removes** AI's response from history (as if it never spoke)
- **Processes** your new message

This prevents the AI from "remembering" responses you interrupted.

### Conversation Context

The client maintains conversation history:
- Last 10 message pairs kept in context
- System prompt always included
- History sent with each LLM request

## Usage Examples

### Basic Conversation

```
[You speak]: "What's the capital of France?"
[Assistant]: "The capital of France is Paris."
[You speak]: "And what's its population?"
[Assistant]: "Paris has a population of about 2.1 million people."
```

### Interrupting Before Response

```
[You speak]: "Tell me about quantum physics"
[LLM] Generating response...
[You speak]: "Actually, make it simple for a 5-year-old"
[Interrupt] Canceling LLM call...
[LLM] Generating response to: "Tell me about quantum physics. Actually, make it simple for a 5-year-old"
```

### Interrupting During Playback

```
[You speak]: "Count to 100"
[Assistant starts]: "One, two, three, four..."
[You speak]: "Stop, count backwards instead"
[Interrupt] Fading out audio...
[Interrupt] Removed assistant's response from history
[LLM] Generating new response to: "Stop, count backwards instead"
```

## Troubleshooting

### LLM Connection Issues

```
[LLM] Error: Cannot connect to localhost:1234
```

**Solution**: Change `llm.host` in config to Windows host IP:
```bash
# Get Windows IP from WSL2
ip route show | grep -i default | awk '{ print $3}'
```

### TTS Connection Issues

```
[TTS] Error: Connection refused
```

**Solution**: Verify TTS server is running and accessible:
```bash
# Test from WSL2
curl -X POST http://localhost:3500/tts-file \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}' \
  --output test.wav
```

### No Audio Playback

```
[Playback] Error: No audio device found
```

**Solution**:
1. Check PulseAudio is running in WSL2:
```bash
pulseaudio --check || pulseaudio --start
```

2. List available devices:
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

3. Set specific device in config:
```yaml
playback:
  output_device: 0  # Device ID from above list
```

### Audio Playback in WSL2

For audio to work in WSL2, you need PulseAudio configured:

1. Install PulseAudio:
```bash
sudo apt-get install pulseaudio
```

2. Start PulseAudio:
```bash
pulseaudio --start
```

3. Test audio:
```bash
speaker-test -t wav -c 2
```

## Command Reference

```bash
# Start Whisper server
python -m src.main_whisper_server

# Start conversational client (default config)
python scripts/conversational_client.py

# Start with custom config
python scripts/conversational_client.py --config my_config.yaml
```

## State Machine

The client uses a state machine to manage flow:

- `LISTENING` - Waiting for user speech
- `WAITING_FOR_TRANSCRIPTION` - VAD detected, waiting for Whisper
- `CALLING_LLM` - Generating LLM response
- `CALLING_TTS` - Synthesizing speech
- `PLAYING_AUDIO` - Playing back audio

Interruptions are handled differently based on current state.
