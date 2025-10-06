"""
Mic capture using sounddevice. Emits 16kHz mono PCM16 frames of frame_ms length.
"""
import sounddevice as sd
import numpy as np
from typing import Iterator
from queue import Queue


class MicStream:
    def __init__(self, sample_rate: int, channels: int, frame_ms: int):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.queue = Queue()
        self.buffer = np.array([], dtype=np.int16)

        # Start the input stream with low latency
        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='float32',
            blocksize=self.frame_samples,
            latency='low',
            callback=self._audio_callback
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        pcm16 = (indata[:, 0] * 32767).astype(np.int16)
        self.queue.put(pcm16)

    def frames(self) -> Iterator[np.ndarray]:
        """
        Yields PCM16 numpy arrays shape=(frame_samples,), dtype=int16.
        """
        while True:
            chunk = self.queue.get()
            self.buffer = np.concatenate([self.buffer, chunk])

            # Yield fixed-size frames
            while len(self.buffer) >= self.frame_samples:
                frame = self.buffer[:self.frame_samples]
                self.buffer = self.buffer[self.frame_samples:]
                yield frame

    def close(self):
        self.stream.stop()
        self.stream.close()
