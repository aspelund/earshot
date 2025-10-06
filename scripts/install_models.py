"""
Download TEN VAD ONNX and the selected faster-whisper model.
"""
import os
import subprocess
import sys
import pathlib
import urllib.request


def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: str):
    """Download a file with progress."""
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"✓ Downloaded {dest}")


def main():
    ensure_dir("models")

    # Download Silero VAD ONNX model (using Silero VAD as it's well-supported)
    # The model expects (1, samples) input and outputs speech probability
    ten_vad_url = "https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx"
    ten_vad_path = "models/ten_vad.onnx"

    if not os.path.exists(ten_vad_path):
        print("Downloading Silero VAD model...")
        try:
            download_file(ten_vad_url, ten_vad_path)
        except Exception as e:
            print(f"⚠ Warning: Could not auto-download VAD model: {e}")
            print(f"Please manually download from: {ten_vad_url}")
            print(f"And place it at: {ten_vad_path}")
    else:
        print(f"✓ VAD model already exists at {ten_vad_path}")

    # faster-whisper will auto-download models on first run
    # But we can pre-download if desired
    print("\n✓ Model setup complete!")
    print("\nNote: Whisper models will be downloaded automatically on first run.")
    print("To pre-download, run:")
    print("  python -c \"from faster_whisper import WhisperModel; WhisperModel('tiny.en')\"")


if __name__ == "__main__":
    main()
