"""Audio playback with support for fade-out and interruption"""
import asyncio
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import sys
from typing import Optional, Tuple


class AudioPlayer:
    """Queue-based audio player with support for fade-out and abort"""

    def __init__(self, fade_out_duration_ms: int = 250, device=None):
        self.default_fade_out_ms = fade_out_duration_ms
        self.device = device

        # Queue for audio files (will be created when loop starts)
        self.queue: Optional[asyncio.Queue] = None

        # Playback state
        self.is_playing = False
        self.should_stop = False
        self.abort_fade_ms: Optional[int] = None

        # Current playback data
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.current_frame = 0
        self.fade_start_frame: Optional[int] = None

        # Playback task (started later)
        self._playback_task: Optional[asyncio.Task] = None
        self._started = False

    def load_wav(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load WAV from bytes, return (audio_data, sample_rate)"""
        with io.BytesIO(wav_bytes) as f:
            data, sr = sf.read(f, dtype='float32')
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        return data, sr

    def start(self) -> None:
        """Start the playback loop (call once inside async context)"""
        if not self._started:
            self.queue = asyncio.Queue()
            self._playback_task = asyncio.create_task(self._playback_loop())
            self._started = True

    def enqueue(self, wav_bytes: bytes) -> None:
        """Add audio to queue (non-blocking). Starts playing immediately if idle."""
        if not self._started:
            raise RuntimeError("AudioPlayer not started. Call start() first in async context.")
        self.queue.put_nowait(wav_bytes)

    def abort(self, fade_ms: int = 100) -> None:
        """
        Clear queue and fade out current playback (non-blocking).
        fade_ms: Duration of fade-out in milliseconds.
        """
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Trigger abort with custom fade duration
        if self.is_playing:
            self.should_stop = True
            self.abort_fade_ms = fade_ms

    async def _playback_loop(self):
        """Background task that continuously plays audio from queue"""
        while True:
            try:
                # Wait for next audio item
                wav_bytes = await self.queue.get()

                # Load and play
                await self._play_audio(wav_bytes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Playback loop error: {e}", file=sys.stderr)

    async def _play_audio(self, wav_bytes: bytes) -> None:
        """Play a single audio file"""
        self.audio_data, self.sample_rate = self.load_wav(wav_bytes)
        self.current_frame = 0
        self.should_stop = False
        self.fade_start_frame = None
        self.abort_fade_ms = None
        self.is_playing = True

        completion_event = asyncio.Event()

        def callback(outdata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}", file=sys.stderr)

            if self.should_stop and self.fade_start_frame is None:
                # Start fade-out
                self.fade_start_frame = self.current_frame

            # Calculate how many frames to output
            remaining = len(self.audio_data) - self.current_frame
            frames_to_output = min(frames, remaining)

            if frames_to_output > 0:
                # Get audio chunk
                chunk = self.audio_data[self.current_frame:self.current_frame + frames_to_output]

                # Apply fade-out if active
                if self.fade_start_frame is not None:
                    # Use abort fade duration if set, otherwise default
                    fade_ms = self.abort_fade_ms if self.abort_fade_ms is not None else self.default_fade_out_ms
                    fade_frames = int(self.sample_rate * fade_ms / 1000)
                    frames_since_fade = self.current_frame - self.fade_start_frame

                    if frames_since_fade < fade_frames:
                        # Apply linear fade
                        fade_curve = np.linspace(
                            1.0 - (frames_since_fade / fade_frames),
                            1.0 - ((frames_since_fade + frames_to_output) / fade_frames),
                            frames_to_output
                        )
                        fade_curve = np.maximum(fade_curve, 0)  # Clamp to 0
                        chunk = chunk * fade_curve
                    else:
                        # Fade complete, stop
                        chunk = np.zeros(frames_to_output)
                        completion_event.set()
                        raise sd.CallbackStop()

                outdata[:frames_to_output] = chunk.reshape(-1, 1)
                self.current_frame += frames_to_output

            # Pad with silence if needed
            if frames_to_output < frames:
                outdata[frames_to_output:] = 0
                completion_event.set()
                raise sd.CallbackStop()

            # Check if we've reached the end
            if self.current_frame >= len(self.audio_data):
                completion_event.set()
                raise sd.CallbackStop()

        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=callback,
                device=self.device
            ):
                await completion_event.wait()
        except Exception as e:
            print(f"Playback error: {e}", file=sys.stderr)
        finally:
            self.is_playing = False

    async def stop(self):
        """Stop the playback loop (cleanup)"""
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
