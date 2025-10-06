#!/usr/bin/env python3
"""
Interactive VAD test - prints speech probabilities in real-time.
Press Ctrl+C to stop.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.audio import MicStream
from src.vad_ten import TenVAD


def main():
    print("Testing VAD...")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mic = MicStream(
        sample_rate=cfg["audio"]["sample_rate"],
        channels=cfg["audio"]["channels"],
        frame_ms=cfg["audio"]["frame_ms"]
    )
    vad = TenVAD(cfg["vad"]["model_path"])

    threshold = cfg["vad"]["threshold"]
    ema = 0.0
    alpha = cfg["vad"]["ema_alpha"]

    try:
        for i, frame in enumerate(mic.frames()):
            # Get VAD probability
            prob = vad.prob_speech(frame)
            ema = alpha * prob + (1 - alpha) * ema

            # Visualize
            bar_len = int(prob * 50)
            ema_len = int(ema * 50)
            bar = "█" * bar_len
            ema_bar = "▓" * ema_len

            speech = "SPEECH" if ema >= threshold else "silence"

            print(f"\rFrame {i:05d} | Raw: {bar:<50} | EMA: {ema_bar:<50} | {speech:<7} (ema={ema:.3f})", end="", flush=True)

            if i > 1000:  # Auto-stop after ~20 seconds
                break

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        mic.close()

    print("✓ VAD test complete!")


if __name__ == "__main__":
    main()
