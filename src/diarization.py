"""
Pyannote speaker diarization wrapper.
Identifies who spoke when in an audio file.
"""
import os
import torch
from typing import List, Dict, Optional


class SpeakerDiarizer:
    """
    Wrapper for Pyannote speaker diarization pipeline.
    Lazy-loads the model on first use.
    """

    def __init__(self, cfg: dict):
        self.model_name = cfg.get("model", "pyannote/speaker-diarization-3.1")
        self.device = cfg.get("device", "cuda")
        self.hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")

        # Lazy load - model is heavy (~1GB)
        self._pipeline = None

    def _load_pipeline(self):
        """Load the diarization pipeline on first use."""
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token
            )

            if self.device == "cuda" and torch.cuda.is_available():
                self._pipeline = self._pipeline.to(torch.device("cuda"))

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict]:
        """
        Run speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers

        Returns:
            List of segments: [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}, ...]
        """
        self._load_pipeline()

        # Build kwargs for optional speaker hints
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        # Run diarization
        result = self._pipeline(audio_path, **kwargs)

        # pyannote.audio >= 3.1 returns DiarizeOutput, access .speaker_diarization
        if hasattr(result, 'speaker_diarization'):
            diarization = result.speaker_diarization
        else:
            diarization = result

        # Convert to list of segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        return segments
