"""
Faster-Whisper wrapper. Transcribes PCM16 bytes into text + word timestamps.
"""
from faster_whisper import WhisperModel
from typing import Dict, Any
import numpy as np


class FastSTT:
    def __init__(self, cfg):
        self.model_size = cfg["model_size"]
        self.compute_type = cfg.get("compute_type", "float16")
        self.device = cfg.get("device", "cuda")  # "cuda" for NVIDIA GPU, "cpu" for CPU

        # Initialize model
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def transcribe(self, pcm16_bytes: bytes, language: str, beam_size: int, word_timestamps: bool) -> Dict[str, Any]:
        """
        Returns dict: {"text": str, "words": [{"w": str, "start_s": float, "end_s": float}, ...]}
        """
        # Convert PCM16 bytes to float32 array normalized to [-1, 1]
        pcm16_array = np.frombuffer(pcm16_bytes, dtype=np.int16)
        audio_float = pcm16_array.astype(np.float32) / 32768.0

        # Transcribe
        segments, info = self.model.transcribe(
            audio_float,
            language=language if language else None,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=False,  # We already did VAD
            condition_on_previous_text=False  # Each segment is independent
        )

        # Collect text and words
        full_text = []
        all_words = []

        for segment in segments:
            full_text.append(segment.text.strip())

            if word_timestamps and segment.words:
                for word in segment.words:
                    all_words.append({
                        "w": word.word.strip(),
                        "start_s": word.start,
                        "end_s": word.end
                    })

        return {
            "text": " ".join(full_text).strip(),
            "words": all_words
        }
