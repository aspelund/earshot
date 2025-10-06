# VAD Server

Local, open-weight, real-time speech pipeline for macOS with multilingual support.

## Features

- **Real-time VAD**: Silero VAD for accurate speech detection with hysteresis
- **Async STT**: faster-whisper on worker thread (never blocks audio)
- **Multilingual**: Supports 99 languages with auto-detection
- **Smart segmentation**: Pre/post padding, hangover, min duration filtering
- **JSONL logging**: Structured logs with UTC timestamps + word-level timing
- **Daemon mode**: Runs as launchd service at login

## Setup on New Machine

### Prerequisites
- macOS (Apple Silicon or Intel)
- Python 3.9+ (`python3 --version`)
- Homebrew (optional, for system dependencies)

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

## Configuration

Edit `config.yaml` to adjust:
- VAD thresholds and timing
- STT model size (tiny.en, base.en, small.en)
- Log rotation settings
- Heartbeat interval

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
