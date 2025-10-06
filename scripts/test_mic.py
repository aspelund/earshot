#!/usr/bin/env python3
"""
Interactive microphone test - prints audio levels.
Press Ctrl+C to stop.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.audio import MicStream


def main():
    print("Testing microphone capture...")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    mic = MicStream(sample_rate=16000, channels=1, frame_ms=20)

    try:
        for i, frame in enumerate(mic.frames()):
            # Calculate RMS level
            rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
            level = int(rms / 100)  # Scale for display
            bar = "█" * min(level, 50)

            # Print level meter
            print(f"\rFrame {i:05d} | Level: {bar:<50} {int(rms):5d}", end="", flush=True)

            if i > 500:  # Auto-stop after ~10 seconds
                break

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        mic.close()

    print("✓ Microphone test complete!")


if __name__ == "__main__":
    main()
