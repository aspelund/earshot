#!/usr/bin/env python3
"""
Simple test client: Record audio in 4-second chunks and send to Whisper server.
No VAD - just time-based segmentation for testing.
"""
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import argparse
import json
import struct
from queue import Queue
from datetime import datetime, timezone


class SimpleTestClient:
    def __init__(self, server_url: str, sample_rate: int = 16000, chunk_seconds: int = 4):
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = sample_rate * chunk_seconds

        self.buffer = np.array([], dtype=np.int16)
        self.audio_queue = Queue(maxsize=500)  # Thread-safe queue

    def audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice - runs in audio thread."""
        if status:
            print(f"Audio status: {status}")

        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        pcm16 = (indata[:, 0] * 32767).astype(np.int16)

        # Put in thread-safe queue (non-blocking)
        try:
            self.audio_queue.put_nowait(pcm16)
        except:
            pass  # Queue full, drop frame

    def encode_segment(self, pcm_bytes: bytes, start_iso: str, end_iso: str) -> bytes:
        """
        Encode segment for transmission: 4-byte header length + JSON header + PCM16 data.
        """
        header = {
            "start_utc": start_iso,
            "end_utc": end_iso
        }
        header_bytes = json.dumps(header).encode('utf-8')
        header_len = len(header_bytes)

        # Pack: uint32 big-endian + header + audio
        return struct.pack('>I', header_len) + header_bytes + pcm_bytes

    async def receive_transcriptions(self, ws):
        """Receive and display transcriptions from server."""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    if data.get("type") == "asr_segment":
                        text = data.get("text", "")
                        latency = data.get("latency_s", 0)
                        print(f"\n[Transcription] {text}")
                        print(f"[Server latency: {latency:.2f}s]\n")
                except json.JSONDecodeError:
                    print(f"Received: {message}")
        except Exception as e:
            print(f"Error receiving: {e}")

    async def run(self):
        """Connect to server and stream audio chunks."""
        print(f"Connecting to {self.server_url}...")

        async with websockets.connect(self.server_url, max_size=10*1024*1024) as ws:
            print(f"Connected!")

            # Start task to receive transcriptions
            receive_task = asyncio.create_task(self.receive_transcriptions(ws))

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=1024,
                callback=self.audio_callback
            ):
                print(f"Recording audio at {self.sample_rate}Hz")
                print(f"Sending {self.chunk_seconds}s chunks to server")
                print("Press Ctrl+C to stop\n")

                try:
                    while True:
                        # Get audio from thread-safe queue (with timeout)
                        await asyncio.sleep(0.01)  # Small delay to yield control

                        # Drain queue
                        while not self.audio_queue.empty():
                            try:
                                chunk = self.audio_queue.get_nowait()
                                self.buffer = np.concatenate([self.buffer, chunk])
                            except:
                                break

                        # Check if we have enough samples for a chunk
                        while len(self.buffer) >= self.chunk_samples:
                            # Extract chunk
                            audio_chunk = self.buffer[:self.chunk_samples]
                            self.buffer = self.buffer[self.chunk_samples:]

                            # Create timestamps
                            now = datetime.now(timezone.utc)
                            duration_ms = int(self.chunk_seconds * 1000)
                            start_dt = datetime.fromtimestamp(
                                now.timestamp() - self.chunk_seconds,
                                tz=timezone.utc
                            )

                            start_iso = start_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                            end_iso = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")

                            # Send to server
                            pcm_bytes = audio_chunk.tobytes()
                            message = self.encode_segment(pcm_bytes, start_iso, end_iso)

                            print(f"[Sending] {self.chunk_seconds}s chunk ({len(pcm_bytes)} bytes)")
                            await ws.send(message)

                except KeyboardInterrupt:
                    print("\nStopping...")
                    receive_task.cancel()


def main():
    parser = argparse.ArgumentParser(description="Simple test client - sends fixed-duration audio chunks")
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="Whisper server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=4,
        help="Chunk duration in seconds (default: 4)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)"
    )

    args = parser.parse_args()

    client = SimpleTestClient(
        server_url=args.server,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nExited")


if __name__ == "__main__":
    main()
