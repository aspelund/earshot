"""
Stateful segmentor that uses VAD probabilities to produce utterance segments with
pre/post padding and hangover. Holds a short ring buffer for pre_ms/post_ms.
"""
from collections import deque
import numpy as np
from typing import Optional, Tuple
from .utils_time import utc_now_iso, now_ms


"""
Stateful segmentor that uses VAD probabilities to produce utterance segments with
pre/post padding and hangover. Holds a short ring buffer for pre_ms/post_ms.
Improvements:
- Hysteresis (start_threshold vs end_threshold)
- Consecutive voiced frames to start (min_start_frames)
- Min speech duration (min_speech_ms)
- True post padding beyond hangover
- Back-compat: works with single `threshold` too
"""
from collections import deque
import numpy as np
from typing import Optional, Tuple
from .utils_time import utc_now_iso, now_ms


class Segmentor:
    def __init__(self, cfg):
        audio_cfg = cfg["audio"]
        vad_cfg = cfg["vad"]

        self.sample_rate   = audio_cfg["sample_rate"]
        self.frame_ms      = audio_cfg["frame_ms"]
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        # Back-compat: if only 'threshold' provided, derive hysteresis
        start_th = vad_cfg.get("start_threshold", vad_cfg.get("threshold", 0.5))
        end_th   = vad_cfg.get("end_threshold", min(0.3, start_th * 0.75))

        self.start_threshold    = float(start_th)
        self.end_threshold      = float(end_th)
        self.ema_alpha          = float(vad_cfg.get("ema_alpha", 0.30))
        self.pre_ms             = int(vad_cfg.get("pre_ms", 200))
        self.end_hang_ms        = int(vad_cfg.get("hang_ms", 250))  # aka hangover
        self.post_ms            = int(vad_cfg.get("post_ms", 200))
        self.max_segment_s      = float(vad_cfg.get("max_segment_s", 20))
        self.min_start_frames   = int(vad_cfg.get("min_start_frames", 3))
        self.min_speech_ms      = int(vad_cfg.get("min_speech_ms", 700))
        self.min_gap_ms         = int(vad_cfg.get("min_gap_ms", 300))  # placeholder for future cross-seg merge

        # Sizes in frames
        self.pre_frames         = max(1, self.pre_ms  // self.frame_ms)
        self.post_frames        = max(1, self.post_ms // self.frame_ms)
        self.hang_frames        = max(1, self.end_hang_ms // self.frame_ms)
        self.max_segment_frames = max(1, int(self.max_segment_s * 1000 // self.frame_ms))

        # State
        self.in_speech            = False
        self.ring_buffer          = deque(maxlen=max(self.pre_frames, self.post_frames))
        self.current_segment      = []
        self.last_voiced_index    = -1  # index into current_segment of last voiced frame
        self.segment_start_ms     = 0
        self.frames_since_start   = 0
        self.consec_voiced        = 0   # consecutive voiced frames required to start

        # EMA smoothed probability
        self.prob_ema = 0.0

    def _ms_to_iso(self, t_ms: int) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)\
                       .isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def update(self, frame_pcm16: np.ndarray, p_speech: float) -> Optional[Tuple[bytes, str, str]]:
        """
        Feed one frame + prob. Returns (segment_pcm16_bytes, start_iso, end_iso) when a segment flushes,
        otherwise None. Handles max_segment_s flushes too.
        """

        # Smooth probability with EMA (fast, low-lag)
        self.prob_ema = self.ema_alpha * p_speech + (1 - self.ema_alpha) * self.prob_ema

        # Feed ring buffer (used for pre/post padding)
        self.ring_buffer.append(frame_pcm16.copy())

        # Determine voiced based on hysteresis (use start_th if idle, end_th if inside)
        th = self.start_threshold if not self.in_speech else self.end_threshold
        is_voiced = (self.prob_ema >= th)

        now_ms_val = now_ms()

        # ---------- not currently inside a segment ----------
        if not self.in_speech:
            # Require a few consecutive voiced frames to start
            if is_voiced:
                self.consec_voiced += 1
            else:
                self.consec_voiced = 0

            if self.consec_voiced >= self.min_start_frames:
                self.in_speech = True
                # Start time is "now minus pre_ms"
                self.segment_start_ms = now_ms_val - (len(self.ring_buffer) * self.frame_ms)
                # Include pre-buffer (up to pre_frames)
                pre = list(self.ring_buffer)[-self.pre_frames:] if self.pre_frames > 0 else []
                self.current_segment = list(pre)
                self.last_voiced_index = len(self.current_segment) - 1 if is_voiced else (len(self.current_segment) - 1)
                self.frames_since_start = len(self.current_segment)
            return None

        # ---------- inside a segment ----------
        self.current_segment.append(frame_pcm16.copy())
        self.frames_since_start += 1

        if is_voiced:
            self.last_voiced_index = len(self.current_segment) - 1

        frames_since_last_voice = (len(self.current_segment) - 1) - self.last_voiced_index

        # Force flush if too long (safety)
        if self.frames_since_start >= self.max_segment_frames:
            return self._flush(reason="max_length", now_ms_val=now_ms_val, extra_post_frames=self.post_frames)

        # End condition: no-voice for hangover duration
        if frames_since_last_voice >= self.hang_frames:
            # We already have 'hangover' silence frames in current_segment.
            # Add true post padding (beyond hangover) if configured.
            extra_post = max(0, self.post_frames - frames_since_last_voice)
            return self._flush(reason="eos", now_ms_val=now_ms_val, extra_post_frames=extra_post)

        return None

    def _flush(self, reason: str, now_ms_val: int, extra_post_frames: int) -> Optional[Tuple[bytes, str, str]]:
        """
        Finalize the current segment:
        - Optionally append a few more frames for post padding beyond hangover.
        - Enforce min_speech_ms.
        - Reset internal state.
        """
        # Append true post padding from ring buffer if needed
        if extra_post_frames > 0 and len(self.ring_buffer) > 0:
            tail = list(self.ring_buffer)[-extra_post_frames:]
            self.current_segment.extend([f.copy() for f in tail])

        # Convert frames to bytes
        if len(self.current_segment) == 0:
            self._reset_after_flush()
            return None

        segment_arr = np.concatenate(self.current_segment)
        segment_bytes = segment_arr.tobytes()

        # Compute times
        start_ms_val = self.segment_start_ms
        # end at "now plus post padding already accounted for by ring frames"
        end_ms_val = start_ms_val + len(self.current_segment) * self.frame_ms

        # Enforce min speech duration (drop tiny blips)
        if (end_ms_val - start_ms_val) < self.min_speech_ms:
            self._reset_after_flush()
            return None

        start_iso = self._ms_to_iso(start_ms_val)
        end_iso   = self._ms_to_iso(end_ms_val)

        # Reset; keep some EMA momentum to avoid immediate re-trigger flicker
        self._reset_after_flush(ema_decay=0.3)

        return (segment_bytes, start_iso, end_iso)

    def _reset_after_flush(self, ema_decay: float = 0.0):
        self.in_speech          = False
        self.current_segment    = []
        self.last_voiced_index  = -1
        self.segment_start_ms   = 0
        self.frames_since_start = 0
        self.consec_voiced      = 0
        if ema_decay:
            self.prob_ema *= float(ema_decay)
        else:
            # do not hard reset EMA; keeping momentum helps continuous speech
            pass
