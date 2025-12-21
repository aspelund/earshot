"""
Audio playback with robust cancellation support.

Improvements over original:
- asyncio.Event for thread-safe cancellation signaling
- Generation counter to filter stale audio chunks
- Atomic abort with queue purging
"""

import asyncio
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import sys
from typing import Optional, Tuple


class AudioPlayer:
    """Queue-based audio player with asyncio.Event cancellation and generation tracking."""

    def __init__(self, fade_out_duration_ms: int = 250, device=None):
        self.default_fade_out_ms = fade_out_duration_ms
        self.device = device

        # Queue for audio chunks
        self.queue: Optional[asyncio.Queue] = None

        # Cancellation via asyncio.Event (thread-safe, waitable)
        self.cancel_event = asyncio.Event()

        # Generation counter for filtering stale chunks
        self.generation = 0
        self.active_generation = 0

        # Playback state
        self.is_playing = False
        self.abort_fade_ms: Optional[int] = None

        # Current playback data
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.current_frame = 0
        self.fade_start_frame: Optional[int] = None

        # Chunk completion tracking
        self.chunks_completed = 0
        self.last_chunk_interrupted = False

        # Background task
        self._playback_task: Optional[asyncio.Task] = None
        self._started = False

    def load_wav(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load WAV from bytes, return (audio_data, sample_rate)."""
        with io.BytesIO(wav_bytes) as f:
            data, sr = sf.read(f, dtype='float32')
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        return data, sr

    def start(self) -> None:
        """Start the playback loop (call once inside async context)."""
        if not self._started:
            self.queue = asyncio.Queue()
            self._playback_task = asyncio.create_task(self._playback_loop())
            self._started = True

    def enqueue(self, wav_bytes: bytes) -> None:
        """
        Add audio to queue (non-blocking).
        Tags the chunk with current generation for filtering.
        """
        if not self._started:
            raise RuntimeError("AudioPlayer not started. Call start() first.")

        # Only enqueue if not cancelled
        if not self.cancel_event.is_set():
            self.queue.put_nowait((wav_bytes, self.generation))

    def abort(self, fade_ms: int = 100) -> None:
        """
        Atomic abort: increment generation, signal cancellation, purge queue.

        Args:
            fade_ms: Duration of fade-out in milliseconds.
        """
        # 1. Increment generation (invalidates all in-flight chunks)
        self.generation += 1

        # 2. Signal cancellation (wakes any waiting coroutines)
        self.cancel_event.set()

        # 3. Purge queue (remove chunks that beat the race condition)
        self.clear_queue()

        # 4. Trigger fade-out on current playback
        if self.is_playing:
            self.abort_fade_ms = fade_ms

    def clear_queue(self) -> None:
        """Clear all pending audio from queue."""
        if self.queue:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def reset_for_new_turn(self) -> None:
        """Reset cancellation state for new conversation turn."""
        self.cancel_event.clear()
        self.active_generation = self.generation

    def reset_tracking(self) -> None:
        """Reset chunk tracking for new response."""
        self.chunks_completed = 0
        self.last_chunk_interrupted = False

    async def _playback_loop(self):
        """Background task that continuously plays audio from queue."""
        while True:
            try:
                # Wait for next audio item with timeout to check cancellation
                try:
                    wav_bytes, chunk_generation = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # Check if chunk is from current generation
                if chunk_generation != self.generation:
                    # Stale chunk, discard silently
                    continue

                # Check cancellation before playing
                if self.cancel_event.is_set():
                    continue

                # Play the chunk
                await self._play_audio(wav_bytes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Playback loop error: {e}", file=sys.stderr)

    async def _play_audio(self, wav_bytes: bytes) -> None:
        """Play a single audio chunk."""
        self.audio_data, self.sample_rate = self.load_wav(wav_bytes)
        self.current_frame = 0
        self.fade_start_frame = None
        self.abort_fade_ms = None
        self.is_playing = True

        completion_event = asyncio.Event()
        was_interrupted = False

        def callback(outdata, frames, time_info, status):
            nonlocal was_interrupted

            if status:
                print(f"Audio status: {status}", file=sys.stderr)

            # Check for abort signal (cancel_event or abort_fade_ms set)
            should_abort = self.cancel_event.is_set() or self.abort_fade_ms is not None

            if should_abort and self.fade_start_frame is None:
                # Start fade-out
                self.fade_start_frame = self.current_frame
                was_interrupted = True

            # Calculate frames to output
            remaining = len(self.audio_data) - self.current_frame
            frames_to_output = min(frames, remaining)

            if frames_to_output > 0:
                chunk = self.audio_data[self.current_frame:self.current_frame + frames_to_output]

                # Apply fade-out if active
                if self.fade_start_frame is not None:
                    fade_ms = self.abort_fade_ms if self.abort_fade_ms else self.default_fade_out_ms
                    fade_frames = int(self.sample_rate * fade_ms / 1000)
                    frames_since_fade = self.current_frame - self.fade_start_frame

                    if frames_since_fade < fade_frames:
                        # Apply linear fade
                        fade_curve = np.linspace(
                            1.0 - (frames_since_fade / fade_frames),
                            1.0 - ((frames_since_fade + frames_to_output) / fade_frames),
                            frames_to_output
                        )
                        fade_curve = np.maximum(fade_curve, 0)
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

            # Check if reached end
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
            # Track completion
            if not was_interrupted:
                self.chunks_completed += 1
                self.last_chunk_interrupted = False
            else:
                self.last_chunk_interrupted = True
            self.is_playing = False

    async def stop(self):
        """Stop the playback loop (cleanup)."""
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
