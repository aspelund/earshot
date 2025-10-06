"""
Network audio stream. Receives PCM16 audio chunks over the network
and provides them as frames, similar to MicStream.
"""
import numpy as np
from typing import Iterator
from queue import Queue
from .audio_base import AudioSource


class NetworkStream(AudioSource):
    """
    Receives audio from network and yields fixed-size frames.
    Thread-safe: audio chunks can be pushed from another thread (e.g., WebSocket handler).
    """

    def __init__(self, sample_rate: int, channels: int, frame_ms: int):
        super().__init__(sample_rate, channels, frame_ms)
        self.queue = Queue()
        self.buffer = np.array([], dtype=np.int16)
        self._closed = False

    def push_audio(self, pcm16_bytes: bytes):
        """
        Push audio data (PCM16 bytes) from network into the stream.
        Called by WebSocket/TCP handler in another thread.
        """
        if self._closed:
            return

        # Convert bytes to int16 array
        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        self.queue.put(pcm16)

    def frames(self) -> Iterator[np.ndarray]:
        """
        Yields PCM16 numpy arrays shape=(frame_samples,), dtype=int16.
        Blocks until enough data is available.
        """
        while not self._closed:
            try:
                chunk = self.queue.get(timeout=1.0)
            except:
                # Timeout - check if closed
                continue

            self.buffer = np.concatenate([self.buffer, chunk])

            # Yield fixed-size frames
            while len(self.buffer) >= self.frame_samples:
                frame = self.buffer[:self.frame_samples]
                self.buffer = self.buffer[self.frame_samples:]
                yield frame

    def close(self):
        """Signal that the stream is closed."""
        self._closed = True
