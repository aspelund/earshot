"""
Merge ASR word timestamps with speaker diarization segments.
Produces speaker-attributed utterances with timestamps.
"""
from typing import List, Dict


def merge_asr_diarization(
    words: List[Dict],
    diarization: List[Dict]
) -> List[Dict]:
    """
    Merge ASR words with diarization segments.

    Args:
        words: ASR output with word timestamps
               [{"w": "hello", "start_s": 0.0, "end_s": 0.5}, ...]
        diarization: Speaker segments from Pyannote
               [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}, ...]

    Returns:
        List of utterances grouped by speaker:
        [{"time": "00:00:00", "speaker": "Speaker 1", "text": "Hello how are you"}, ...]
    """
    if not words:
        return []

    if not diarization:
        # No diarization - return all as single speaker
        return [{
            "time": format_timestamp(words[0].get("start_s", 0)),
            "speaker": "Speaker 1",
            "text": " ".join(w["w"] for w in words).strip()
        }]

    # Build speaker map for friendly names (SPEAKER_00 -> Speaker 1)
    speakers = sorted(set(d["speaker"] for d in diarization))
    speaker_map = {s: f"Speaker {i + 1}" for i, s in enumerate(speakers)}

    def find_speaker(word_start: float, word_end: float) -> str:
        """Find speaker with maximum overlap for a word."""
        best_overlap = 0
        best_speaker = None

        for seg in diarization:
            overlap_start = max(word_start, seg["start"])
            overlap_end = min(word_end, seg["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]

        return speaker_map.get(best_speaker, "Unknown")

    # Group words into utterances by speaker
    utterances = []
    current_speaker = None
    current_words = []
    current_start = None

    for word in words:
        word_start = word.get("start_s", 0)
        word_end = word.get("end_s", word_start)
        speaker = find_speaker(word_start, word_end)

        if speaker != current_speaker:
            # Save previous utterance
            if current_words:
                utterances.append({
                    "time": format_timestamp(current_start),
                    "speaker": current_speaker,
                    "text": " ".join(current_words).strip()
                })
            # Start new utterance
            current_speaker = speaker
            current_words = [word["w"]]
            current_start = word_start
        else:
            current_words.append(word["w"])

    # Don't forget last utterance
    if current_words:
        utterances.append({
            "time": format_timestamp(current_start),
            "speaker": current_speaker,
            "text": " ".join(current_words).strip()
        })

    return utterances


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    if seconds is None:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript(utterances: List[Dict]) -> str:
    """
    Format utterances as plain text output.

    Args:
        utterances: List from merge_asr_diarization()

    Returns:
        Formatted string like:
        00:00:00 Speaker 1: Hello, how are you today?
        00:00:04 Speaker 2: I'm doing well, thank you for asking.
    """
    lines = []
    for u in utterances:
        lines.append(f"{u['time']} {u['speaker']}: {u['text']}")
    return "\n".join(lines)
