"""
Base class for audio sources. Provides a common interface for microphone
capture, network streams, or file playback.
"""
from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np


class AudioSource(ABC):
    """Abstract base class for audio input sources."""

    def __init__(self, sample_rate: int, channels: int, frame_ms: int):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """
        Yields PCM16 numpy arrays shape=(frame_samples,), dtype=int16.
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
