#!/usr/bin/env python3
"""
Audio streaming client - captures microphone and streams to server.
Run this on a machine with a microphone (e.g., laptop, phone).
"""
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import argparse
import sys


class AudioStreamClient:
    def __init__(self, server_url: str, sample_rate: int = 16000, frame_ms: int = 30, auth_token: str = None):
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.auth_token = auth_token

    def audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice - runs in audio thread."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        pcm16 = (indata[:, 0] * 32767).astype(np.int16)

        # Put in async queue (non-blocking)
        try:
            self.audio_queue.put_nowait(pcm16.tobytes())
        except:
            pass  # Queue full, drop frame

    async def stream_audio(self):
        """Connect to server and stream microphone audio."""
        print(f"Connecting to {self.server_url}...")

        async with websockets.connect(self.server_url) as ws:
            print(f"Connected to server")

            # Send authentication token if required
            if self.auth_token:
                await ws.send(self.auth_token)
                print("Sent authentication token")

            # Start microphone capture
            self.audio_queue = asyncio.Queue(maxsize=100)

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.frame_samples,
                callback=self.audio_callback
            ):
                print(f"Streaming audio at {self.sample_rate}Hz, {self.frame_ms}ms frames...")
                print("Press Ctrl+C to stop")

                try:
                    while True:
                        # Get audio chunk from queue
                        audio_bytes = await self.audio_queue.get()

                        # Send to server
                        await ws.send(audio_bytes)

                except KeyboardInterrupt:
                    print("\nStopping...")


def main():
    parser = argparse.ArgumentParser(description="Stream microphone audio to server")
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=30,
        help="Frame duration in milliseconds (default: 30)"
    )
    parser.add_argument(
        "--auth-token",
        help="Optional authentication token"
    )

    args = parser.parse_args()

    client = AudioStreamClient(
        server_url=args.server,
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        auth_token=args.auth_token
    )

    try:
        asyncio.run(client.stream_audio())
    except KeyboardInterrupt:
        print("\nExited")


if __name__ == "__main__":
    main()
