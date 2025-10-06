# Testing Guide

This document describes how to test each milestone of the STT daemon.

## Prerequisites

```bash
source .venv/bin/activate
python scripts/install_models.py
```

## M0: Bootstrap Testing

### Automated Tests
```bash
# Verify dependencies
pip list | grep -E "sounddevice|numpy|faster-whisper"

# Run basic imports
python -c "import src.utils_time; print('✓ Imports work')"
```

### Manual Tests
- [ ] Virtual environment created successfully
- [ ] All dependencies installed without errors
- [ ] README is clear and complete
- [ ] Config file is valid YAML

**Expected:** No errors, all packages listed.

---

## M1: Audio + VAD Loop Testing

### Automated Tests
```bash
pytest tests/test_utils_time.py -v
pytest tests/test_segmentor.py -v
```

### Manual Tests

#### Test 1: Microphone Access
```bash
python scripts/test_mic.py
```
- [ ] Microphone permission granted
- [ ] Audio levels respond to speech
- [ ] No errors or warnings

**Expected:** Visual level meter shows bars when speaking.

#### Test 2: VAD Detection
```bash
python scripts/test_vad.py
```
- [ ] "SPEECH" appears when speaking
- [ ] "silence" appears when quiet
- [ ] EMA smooths out noise spikes

**Expected:** Accurate speech/silence detection with ~0.5s response time.

#### Test 3: Sensitivity Tuning
Edit `config.yaml`:
```yaml
vad:
  threshold: 0.55  # Increase for fewer false positives
```
Run `python scripts/test_vad.py` again.

- [ ] Higher threshold requires louder speech
- [ ] Lower threshold (0.45) more sensitive

**Expected:** Threshold affects detection sensitivity.

#### Test 4: False Positive Check
Run `python scripts/test_vad.py` in silence with background noise (music, fan, etc.)

- [ ] No false "SPEECH" triggers during silence
- [ ] Occasional triggers are smoothed by EMA

**Expected:** < 1 false positive per minute.

---

## M2: STT Integration Testing

### Automated Tests
```bash
# Test will be added after implementation
pytest tests/test_stt.py -v
```

### Manual Tests

#### Test 1: Basic Transcription
```bash
bash scripts/dev_run.sh
```
Speak clearly: "Hello world, this is a test."

- [ ] Transcription appears in console
- [ ] Text is accurate
- [ ] No crashes or errors

**Expected:** Correct transcription within 1-2 seconds.

#### Test 2: Word Timestamps
Check console output for word timestamps.

- [ ] Timestamps are absolute UTC (milliseconds since epoch)
- [ ] Timestamps increase monotonically
- [ ] Word order matches text

**Expected:** Each word has `start_ms` and `end_ms` in UTC.

#### Test 3: Fast Speech
Speak rapidly: "One two three four five six seven eight nine ten."

- [ ] All words captured
- [ ] No word chopping
- [ ] Latency still reasonable

**Expected:** Complete transcription, latency < 0.5s.

#### Test 4: Latency Measurement
Watch console for `latency_s` field.

- [ ] `tiny.en`: < 0.3s
- [ ] `base.en`: < 0.5s

**Expected:** Meets latency targets.

#### Test 5: Accent/Style Testing
Try different speaking styles:
- Whisper
- Loud voice
- Different accents
- Background noise

- [ ] Reasonable accuracy across styles
- [ ] No crashes

**Expected:** Degrades gracefully, no crashes.

---

## M3: Logging + Daemon Testing

### Manual Tests

#### Test 1: JSONL Tailing
Terminal 1:
```bash
bash scripts/dev_run.sh
```

Terminal 2:
```bash
tail -f ~/stt_logs/asr.jsonl
```

Speak: "Testing JSONL output."

- [ ] JSONL appears in tail output
- [ ] Each line is valid JSON
- [ ] Schema matches spec

**Expected:** Real-time JSONL output.

#### Test 2: Schema Validation
```bash
cat ~/stt_logs/asr.jsonl | jq .
```

- [ ] All fields present: `type`, `start_utc`, `end_utc`, `latency_s`, `text`, `tokens`
- [ ] `tokens` array has `w`, `start_ms`, `end_ms`
- [ ] No JSON parse errors

**Expected:** Valid JSON, correct schema.

#### Test 3: Heartbeat
Let daemon run for 2+ minutes.

```bash
grep heartbeat ~/stt_logs/asr.jsonl
```

- [ ] Heartbeat appears every ~60 seconds
- [ ] Has `type: "heartbeat"` and `ts` fields

**Expected:** Regular heartbeats.

#### Test 4: Log Rotation
Generate >10MB of logs (speak continuously or adjust `rotate_mb` to 1 for testing).

```bash
ls -lh ~/stt_logs/
```

- [ ] Multiple log files appear (asr.jsonl, asr.jsonl.1, etc.)
- [ ] Files respect size limit
- [ ] Old files are backed up

**Expected:** Rotation occurs, backups created.

#### Test 5: Launchd Start
```bash
cp launchd/com.local.stt.daemon.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.local.stt.daemon.plist
launchctl start com.local.stt.daemon

# Check if running
launchctl list | grep stt.daemon

# Check logs
tail -f /tmp/stt-daemon.err
tail -f ~/stt_logs/asr.jsonl
```

- [ ] Daemon starts without errors
- [ ] Logs appear in ~/stt_logs/
- [ ] No errors in /tmp/stt-daemon.err

**Expected:** Daemon runs in background.

#### Test 6: Auto-Start on Reboot
```bash
# Daemon should already be loaded from Test 5
# Reboot your Mac
# After reboot, check:
launchctl list | grep stt.daemon
tail ~/stt_logs/asr.jsonl
```

- [ ] Daemon auto-started
- [ ] New logs appear after reboot

**Expected:** Daemon runs automatically after login.

---

## M4: Validation & Tuning Testing

### Manual Tests

#### Test 1: Short Bursts
Speak short phrases with pauses:
- "Hello"
- [pause 2s]
- "World"
- [pause 2s]
- "Test"

- [ ] Each phrase is a separate segment
- [ ] Fast start (< 0.5s delay)
- [ ] Clean end (no word chopping)

**Expected:** Crisp segmentation.

#### Test 2: Hangover Tuning
Edit `config.yaml`:
```yaml
vad:
  hang_ms: 300  # Try different values: 200, 250, 300, 350
```

Speak with natural pauses: "This is... a test... with pauses."

- [ ] 200ms: May chop mid-sentence pauses
- [ ] 300ms: Better for natural speech
- [ ] 350ms: May merge separate utterances

**Expected:** Find optimal value for your speaking style.

#### Test 3: Threshold Tuning
Edit `config.yaml`:
```yaml
vad:
  threshold: 0.45  # Try: 0.45, 0.50, 0.55, 0.60
```

- [ ] 0.45: More sensitive, more false positives
- [ ] 0.55: Less sensitive, may miss quiet speech

**Expected:** Balance sensitivity vs. false positives.

#### Test 4: Model Comparison
Edit `config.yaml`:
```yaml
stt:
  model_size: "base.en"  # Try: tiny.en, base.en, small.en
```

Speak challenging phrase: "The quick brown fox jumps over the lazy dog."

- [ ] `tiny.en`: Fast, lower accuracy
- [ ] `base.en`: Good balance
- [ ] `small.en`: Best accuracy, slower

**Expected:** Accuracy improves with larger models, latency increases.

#### Test 5: Edge Cases
- **Whispering**: Speak very quietly
- **Shouting**: Speak loudly
- **Fast speech**: Speak as fast as possible
- **Long monologue**: Speak continuously for 30+ seconds

- [ ] System handles all cases gracefully
- [ ] No crashes or hangs
- [ ] Quality degrades but doesn't break

**Expected:** Robust handling of edge cases.

#### Test 6: CPU Monitoring
```bash
# In one terminal
bash scripts/dev_run.sh

# In another
top -pid $(pgrep -f "src.main")
```

Speak continuously for 1 minute.

- [ ] CPU usage reasonable (< 50% on average)
- [ ] No memory leaks
- [ ] Latency stable

**Expected:** Sustainable resource usage.

---

## Acceptance Criteria Summary

### M0 ✓
- All dependencies install
- Basic imports work
- Config valid

### M1 ✓
- Mic capture works
- VAD detects speech accurately
- Segmentor produces clean segments
- < 1 false positive per minute

### M2 ✓
- Transcription accurate (>90% WER for clear speech)
- Word timestamps in UTC
- Latency < 0.3s for tiny.en, < 0.5s for base.en

### M3 ✓
- JSONL logs valid and real-time
- Rotation works
- Heartbeat every 60s
- Daemon auto-starts

### M4 ✓
- Tuning parameters affect behavior as expected
- System handles edge cases
- CPU usage sustainable
- Documentation complete

---

## Troubleshooting

### No microphone input
- Check System Settings > Privacy & Security > Microphone
- Grant permission to Terminal or your IDE

### VAD model not found
- Run `python scripts/install_models.py`
- Check `models/ten_vad.onnx` exists

### High latency
- Use smaller model: `tiny.en`
- Reduce `beam_size` to 1
- Check CPU isn't throttling

### False positives
- Increase `vad.threshold` to 0.55
- Increase `ema_alpha` to 0.4 for more smoothing

### Chopped words
- Increase `hang_ms` to 300-350
- Increase `post_ms` to 250

### Daemon won't start
- Check `/tmp/stt-daemon.err` for errors
- Verify paths in plist file
- Ensure venv exists and has dependencies
