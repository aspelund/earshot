#!/usr/bin/env python3
"""
Record audio and test STT directly (bypassing VAD).
Press Ctrl+C to stop recording and transcribe.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from src.audio import MicStream
from src.stt_whisper import FastSTT
import time


def main():
    print("STT Direct Test")
    print("=" * 60)
    print("Recording will start in 2 seconds...")
    print("Speak clearly, then press Ctrl+C to stop and transcribe.\n")

    time.sleep(2)

    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize components
    mic = MicStream(
        sample_rate=cfg["audio"]["sample_rate"],
        channels=cfg["audio"]["channels"],
        frame_ms=cfg["audio"]["frame_ms"]
    )
    stt = FastSTT(cfg["stt"])

    # Record frames
    frames = []
    print("🔴 RECORDING... (Ctrl+C to stop)")

    try:
        for i, frame in enumerate(mic.frames()):
            frames.append(frame)
            # Show visual progress
            if i % 50 == 0:  # Every second
                print(f"  {i // 50}s...", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n⏹  Stopped recording.")
    finally:
        mic.close()

    if not frames:
        print("❌ No audio recorded!")
        return

    # Convert to bytes
    audio_array = np.concatenate(frames)
    audio_bytes = audio_array.tobytes()

    duration = len(frames) * cfg["audio"]["frame_ms"] / 1000
    print(f"📊 Recorded {duration:.1f}s of audio ({len(audio_bytes)} bytes)")

    # Transcribe
    print("\n🔄 Transcribing...")
    t0 = time.perf_counter()
    result = stt.transcribe(
        audio_bytes,
        language=cfg["stt"]["language"],
        beam_size=cfg["stt"]["beam_size"],
        word_timestamps=cfg["stt"]["word_timestamps"]
    )
    latency = time.perf_counter() - t0

    # Display results
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT:")
    print("=" * 60)
    print(f"Text: {result['text']}")
    print(f"Latency: {latency:.3f}s")
    print(f"Real-time factor: {latency/duration:.3f}x")

    if result['words']:
        print(f"\nWords ({len(result['words'])}):")
        for i, word in enumerate(result['words'], 1):
            print(f"  {i:2d}. [{word['start_s']:6.2f}s - {word['end_s']:6.2f}s] {word['w']}")
    else:
        print("\n(No word timestamps)")

    print("=" * 60)


if __name__ == "__main__":
    main()
