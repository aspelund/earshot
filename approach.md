# Background (for Codex)

We are building a **local, open-weight, real-time speech pipeline** on macOS that:

1. Captures microphone audio (16 kHz mono).
2. Runs **TEN VAD** for frame-level speech detection (fast start/end of speech).
3. Segments audio with **pre/post padding + hangover** to avoid chopped words.
4. Transcribes each segment with a **fast Whisper variant** (faster-whisper / CTranslate2) using a tiny/base model for low latency.
5. Emits **JSONL logs** (one JSON per line) with absolute **UTC timestamps**, text, and word timings.
6. Runs as a **launchd** user daemon at login; logs rotate.

Constraints: offline, open-weight models, real-time on a normal Mac (Apple Silicon or Intel). Downstream agents will `tail -f` the JSONL files.

---

# Milestones (with clear outputs)

**M0. Bootstrap**

- Python venv, dependencies, minimal README, `requirements.txt`.

**M1. Audio + VAD Loop**

- Mic capture @ 16 kHz, 20 ms frames.
- TEN VAD probability per frame.
- Start/end detection: `pre_ms`, `hang_ms`, `post_ms`, `vad_thresh`.

**M2. STT Integration**

- Segment flush triggers faster-whisper transcription (greedy / beam=1, int8).
- Word timestamps normalized to absolute UTC.

**M3. Logging + Daemon**

- Rotating **JSONL** logs in `~/stt_logs`.
- `launchd` plist to auto-start at login.
- Health heartbeat line every 60 s.

**M4. Validation & Tuning**

- CLI demo, latency + real-time factor prints.
- Knobs for thresholds, model sizes, language.

---

# Target file/folder layout

```
stt-daemon/
  README.md
  requirements.txt
  .env.example
  config.yaml
  src/
    audio.py
    vad_ten.py
    stt_whisper.py
    segmentor.py
    logger.py
    main.py
    utils_time.py
  scripts/
    install_models.py
    dev_run.sh
  launchd/
    com.local.stt.daemon.plist
  tests/
    test_segmentor.py
    fixtures/
  stt_logs/            # created at runtime
```

---

# Dependencies (pin lightly)

`requirements.txt`

```
sounddevice>=0.4.6
numpy>=1.26
PyYAML>=6.0
loguru>=0.7
python-dotenv>=1.0
webrtcvad>=2.0.10           # optional fallback gate
onnxruntime>=1.18.0         # for TEN VAD ONNX inference
ctranslate2>=4.4.0          # faster-whisper engine
faster-whisper>=1.0.0
```

> Optional: if you prefer sherpa-onnx bindings for TEN VAD, add it and remove raw onnxruntime calls.

---

# Config (central knobs)

`config.yaml`

```yaml
audio:
  sample_rate: 16000
  channels: 1
  frame_ms: 20

vad:
  threshold: 0.50 # speech prob threshold
  ema_alpha: 0.30 # smoothing for stability
  pre_ms: 200 # prepend before speech start
  hang_ms: 250 # no-voice hangover before EOS
  post_ms: 200 # tail after EOS
  max_segment_s: 20 # force flush long segments
  model_path: "./models/ten_vad.onnx"

stt:
  model_size: "tiny.en" # tiny.en | base.en | small.en
  compute_type: "int8" # int8 | int8_float16 | float16
  language: "en" # or null for auto
  beam_size: 1 # 1 = greedy for speed
  word_timestamps: true

logging:
  dir: "~/stt_logs"
  file: "asr.jsonl"
  rotate_mb: 10
  backups: 5
  heartbeat_s: 60
```

---

# Commands (developer loop)

```bash
# M0: bootstrap
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# M0: set env
cp .env.example .env
# (Optional) export overrides here

# M0: get models
python scripts/install_models.py

# M1–M3: run dev
bash scripts/dev_run.sh

# Tail logs
tail -f ~/stt_logs/asr.jsonl
```

---

# .env example (overrides)

`.env.example`

```
CONFIG_PATH=config.yaml
LOG_DIR=~/stt_logs
```

---

# Meta-code stubs (Codex should fill these out)

## `src/audio.py`

```python
"""
Mic capture using sounddevice. Emits 16kHz mono PCM16 frames of frame_ms length.
"""
import sounddevice as sd
import numpy as np
from typing import Iterator

class MicStream:
    def __init__(self, sample_rate: int, channels: int, frame_ms: int):
        # TODO: init input stream with low latency
        pass

    def frames(self) -> Iterator[np.ndarray]:
        """
        Yields PCM16 numpy arrays shape=(samples,), dtype=int16 of size frame_samples.
        """
        # TODO: implement callback->queue, convert float32 to int16, yield fixed-size frames
        pass
```

## `src/vad_ten.py`

```python
"""
TEN VAD: ONNX runtime session with a simple feature pipeline and prob_speech(frame)->float.
Assumes 16kHz PCM16, 20ms frames. Apply EMA smoothing in caller if desired.
"""
import onnxruntime as ort
import numpy as np

class TenVAD:
    def __init__(self, model_path: str):
        # TODO: ort.InferenceSession with optimization EPs (CPU / CoreML if available)
        pass

    def reset(self):
        # TODO: reset internal states (if model uses context)
        pass

    def prob_speech(self, pcm16: np.ndarray) -> float:
        """
        :param pcm16: int16 mono frame of length frame_samples
        :return: float probability 0..1
        """
        # TODO: feature extraction (e.g., log-mel) if required by the TEN model
        # TODO: run ONNX session, return probability
        pass
```

## `src/segmentor.py`

```python
"""
Stateful segmentor that uses VAD probabilities to produce utterance segments with
pre/post padding and hangover. Holds a short ring buffer for pre_ms/post_ms.
"""
from collections import deque
import numpy as np
from typing import Optional, Iterable, Tuple
from .utils_time import utc_now_iso, now_ms

class Segmentor:
    def __init__(self, cfg):
        # TODO: init thresholds, durations, ring buffer sized to pre_ms/post_ms
        pass

    def update(self, frame_pcm16: np.ndarray, p_speech: float) -> Optional[Tuple[bytes, str, str]]:
        """
        Feed one frame + prob. Returns (segment_pcm16_bytes, start_iso, end_iso) when a segment flushes,
        otherwise None. Handles max_segment_s flushes too.
        """
        # TODO: implement in_speech state, last_voiced_ts, pre/post padding, hangover
        pass
```

## `src/stt_whisper.py`

```python
"""
Faster-Whisper wrapper. Transcribes PCM16 bytes into text + word timestamps.
"""
from faster_whisper import WhisperModel
from typing import Dict, Any

class FastSTT:
    def __init__(self, cfg):
        # TODO: model = WhisperModel(cfg.model_size, compute_type=cfg.compute_type)
        pass

    def transcribe(self, pcm16_bytes: bytes, language: str, beam_size: int, word_timestamps: bool) -> Dict[str, Any]:
        """
        Returns dict: {"text": str, "words": [{"w": str, "start_s": float, "end_s": float}, ...]}
        """
        # TODO: convert bytes->float32 array, call model.transcribe, collect words
        pass
```

## `src/logger.py`

```python
"""
Rotating JSONL logger + heartbeat. One JSON per line for downstream agents.
"""
from loguru import logger
import json, os, time
from typing import Dict

class JSONLLogger:
    def __init__(self, path: str, rotate_mb: int, backups: int, heartbeat_s: int):
        # TODO: configure loguru rotation, set heartbeat timer
        pass

    def append_segment(self, seg: Dict):
        # seg contains start_utc/end_utc/text/words/latency etc.
        # TODO: logger.info(json.dumps(seg, ensure_ascii=False))
        pass

    def heartbeat(self):
        # TODO: write {"type":"heartbeat","ts":...}
        pass
```

## `src/utils_time.py`

```python
from datetime import datetime, timezone

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")

def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
```

## `src/main.py`

```python
"""
Wire: Mic -> TEN VAD -> Segmentor -> STT -> JSONL logger (+ heartbeat).
All real-time work must not block the audio callback.
"""
import os, yaml, time, threading
from dotenv import load_dotenv
from .audio import MicStream
from .vad_ten import TenVAD
from .segmentor import Segmentor
from .stt_whisper import FastSTT
from .logger import JSONLLogger
from .utils_time import utc_now_iso

def load_cfg():
    path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    load_dotenv()
    cfg = load_cfg()

    # Ensure log dir exists
    log_dir = os.path.expanduser(cfg["logging"]["dir"])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, cfg["logging"]["file"])

    # Init components
    mic = MicStream(cfg["audio"]["sample_rate"], cfg["audio"]["channels"], cfg["audio"]["frame_ms"])
    vad = TenVAD(cfg["vad"]["model_path"])
    seg = Segmentor(cfg)
    stt = FastSTT(cfg["stt"])
    jl  = JSONLLogger(log_path, cfg["logging"]["rotate_mb"], cfg["logging"]["backups"], cfg["logging"]["heartbeat_s"])

    # Heartbeat thread
    def heart():
        while True:
            time.sleep(cfg["logging"]["heartbeat_s"])
            jl.heartbeat()
    threading.Thread(target=heart, daemon=True).start()

    # Main loop
    for frame in mic.frames():
        p = vad.prob_speech(frame)  # float 0..1
        flush = seg.update(frame, p)
        if flush:
            pcm_bytes, start_iso, end_iso = flush
            t0 = time.perf_counter()
            out = stt.transcribe(
                pcm_bytes,
                language=cfg["stt"]["language"],
                beam_size=cfg["stt"]["beam_size"],
                word_timestamps=cfg["stt"]["word_timestamps"]
            )
            latency = round(time.perf_counter() - t0, 3)
            seg_json = {
                "type": "asr_segment",
                "start_utc": start_iso,
                "end_utc": end_iso,
                "latency_s": latency,
                "text": out["text"],
                "tokens": [
                    {"w": w["w"], "start_ms": w["start_ms"], "end_ms": w["end_ms"]}
                    for w in out["words"]
                ]
            }
            jl.append_segment(seg_json)

if __name__ == "__main__":
    main()
```

---

# Model download helper

`scripts/install_models.py`

```python
"""
Download TEN VAD ONNX and the selected faster-whisper model.
"""
import os, subprocess, sys, pathlib

def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def main():
    ensure_dir("models")
    # TODO: download TEN VAD ONNX to ./models/ten_vad.onnx (from the official release URL)
    # TODO: optionally verify hash

    # faster-whisper pulls models on first run; optionally prefetch using ctranslate2
    # Example (optional): subprocess.run([...]) to pre-download tiny.en

if __name__ == "__main__":
    main()
```

---

# Dev runner

`scripts/dev_run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export CONFIG_PATH=${CONFIG_PATH:-config.yaml}
python -m stt-daemon.src.main
```

---

# launchd (auto-start at login)

`launchd/com.local.stt.daemon.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.local.stt.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd ~/stt-daemon && source .venv/bin/activate && bash scripts/dev_run.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/stt-daemon.out</string>
  <key>StandardErrorPath</key><string>/tmp/stt-daemon.err</string>
</dict></plist>
```

Load it:

```bash
launchctl load ~/stt-daemon/launchd/com.local.stt.daemon.plist
launchctl start com.local.stt.daemon
```

---

# Log schema (JSONL; one line per segment)

```json
{
  "type": "asr_segment",
  "start_utc": "2025-10-06T07:59:12.184Z",
  "end_utc":   "2025-10-06T07:59:13.902Z",
  "latency_s": 0.17,
  "text": "okay let's kick off",
  "tokens": [
    { "w":"okay", "start_ms":1696582752184, "end_ms":1696582752500 },
    { "w":"let's", "start_ms":..., "end_ms":... }
  ]
}
```

> Your agent can `tail -f ~/stt_logs/asr.jsonl`.

---

# Validation checklist

- Speak short bursts. Confirm **fast start** and **clean end** (no chopped suffix).
- Check `latency_s` < 0.3 s for tiny/base models.
- Toggle `vad.threshold` (0.45–0.55) and `hang_ms` (200–350) to optimize in your room.
- Try background music; ensure low false triggers.
- Increase `stt.model_size` to `base.en` if accuracy feels too low.

---

# Performance knobs (quick wins)

- **Lower latency:** `post_ms` 150–200, `hang_ms` 200–250, `beam_size=1`.
- **Fewer false starts:** raise `vad.threshold` to 0.55 and add EMA (`ema_alpha≈0.3`).
- **Accuracy bump:** switch to `base.en` or `small.en`; keep `compute_type=int8`.
- **CPU footprint:** keep `frame_ms=20`; STT on a worker thread if needed.
