#!/usr/bin/env python3
"""
VAD Client: Captures microphone, runs VAD+segmentation locally,
sends complete segments to Whisper server for transcription.
"""
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import argparse
import sys
import json
import struct
import yaml
import os
from queue import Queue
from datetime import datetime, timezone


class VADClient:
    def __init__(self, config_path: str, server_url: str, auth_token: str = None):
        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.server_url = server_url
        self.auth_token = auth_token

        # Audio config
        self.sample_rate = self.cfg["audio"]["sample_rate"]
        self.frame_ms = self.cfg["audio"]["frame_ms"]
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        # Import VAD and Segmentor (need to add parent dir to path)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.vad_ten import TenVAD
        from src.segmentor import Segmentor

        # Initialize VAD and Segmentor
        print("Loading VAD model...")
        self.vad = TenVAD(self.cfg["vad"]["model_path"])
        self.segmentor = Segmentor(self.cfg)
        print("VAD initialized")

        # Audio queue (thread-safe)
        self.audio_queue = Queue(maxsize=500)

        # Stats
        self.frames_received = 0
        self.frames_dropped = 0
        self.segments_sent = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice - runs in audio thread."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        pcm16 = (indata[:, 0] * 32767).astype(np.int16)

        # Put in async queue (non-blocking)
        try:
            self.audio_queue.put_nowait(pcm16)
            self.frames_received += 1
        except:
            self.frames_dropped += 1
            if self.frames_dropped % 100 == 0:
                print(f"⚠ Dropped {self.frames_dropped} frames (queue full)", file=sys.stderr)

    async def receive_transcriptions(self, ws):
        """Receive and display transcriptions from server."""
        try:
            async for message in ws:
                # Parse JSON transcription result
                try:
                    data = json.loads(message)
                    if data.get("type") == "asr_segment":
                        # Display transcription
                        text = data.get("text", "")
                        latency = data.get("latency_s", 0)
                        print(f"\n[Transcription] {text}")
                        print(f"[Server latency: {latency:.2f}s]")
                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}")
        except Exception as e:
            print(f"Error receiving transcriptions: {e}")

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

    async def run(self):
        """Connect to server and stream VAD-segmented audio."""
        print(f"Connecting to {self.server_url}...")

        async with websockets.connect(self.server_url, max_size=10*1024*1024) as ws:
            print(f"Connected to Whisper server")

            # Send authentication token if required
            if self.auth_token:
                await ws.send(self.auth_token)
                print("Sent authentication token")

            # Start task to receive transcriptions
            receive_task = asyncio.create_task(self.receive_transcriptions(ws))

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.frame_samples,
                callback=self.audio_callback
            ):
                print(f"Listening on microphone ({self.sample_rate}Hz, {self.frame_ms}ms frames)")
                print("Running VAD locally, sending segments to server...")
                print("Press Ctrl+C to stop\n")

                try:
                    last_stats_time = asyncio.get_event_loop().time()
                    last_heartbeat_time = asyncio.get_event_loop().time()
                    loop_iterations = 0

                    while True:
                        loop_iterations += 1
                        # Small delay to yield control
                        await asyncio.sleep(0.001)

                        # Drain thread-safe queue
                        frames = []
                        while not self.audio_queue.empty():
                            try:
                                frames.append(self.audio_queue.get_nowait())
                            except:
                                break

                        # Process all frames
                        for frame in frames:
                            # Run VAD
                            prob = self.vad.prob_speech(frame)

                            # Update segmentor
                            result = self.segmentor.update(frame, prob)

                            if result:
                                # Segment complete - send to server
                                pcm_bytes, start_iso, end_iso = result

                                # Calculate segment duration
                                duration_s = len(pcm_bytes) / 2 / self.sample_rate
                                self.segments_sent += 1
                                print(f"\n[VAD] Segment #{self.segments_sent} detected ({duration_s:.2f}s, prob={prob:.2f})")
                                print(f"[VAD] Encoding and sending to server...")

                                # Encode and send
                                message = self.encode_segment(pcm_bytes, start_iso, end_iso)

                                try:
                                    send_start = asyncio.get_event_loop().time()
                                    await asyncio.wait_for(ws.send(message), timeout=5.0)
                                    send_time = asyncio.get_event_loop().time() - send_start
                                    print(f"[VAD] Sent to server (took {send_time:.3f}s)")
                                except asyncio.TimeoutError:
                                    print(f"⚠ Timeout sending segment to server!", file=sys.stderr)
                                except Exception as e:
                                    print(f"⚠ Error sending segment: {e}", file=sys.stderr)

                        # Print heartbeat every 2 seconds (to show loop is alive)
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_heartbeat_time > 2.0:
                            in_speech = self.segmentor.in_speech
                            queue_size = self.audio_queue.qsize()
                            print(f"[Heartbeat] Loop alive, queue: {queue_size}, in_speech: {in_speech}, iterations: {loop_iterations}")
                            last_heartbeat_time = current_time
                            loop_iterations = 0

                        # Print stats every 10 seconds
                        if current_time - last_stats_time > 10.0:
                            queue_size = self.audio_queue.qsize()
                            print(f"[Stats] Frames: {self.frames_received}, Dropped: {self.frames_dropped}, "
                                  f"Segments sent: {self.segments_sent}, Queue: {queue_size}")
                            last_stats_time = current_time

                except KeyboardInterrupt:
                    print("\nStopping...")
                    receive_task.cancel()


def main():
    parser = argparse.ArgumentParser(description="VAD client - runs VAD locally, sends segments to Whisper server")
    parser.add_argument(
        "--config",
        default="config.vad_client.yaml",
        help="Config file path (default: config.vad_client.yaml)"
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="Whisper server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--auth-token",
        help="Optional authentication token"
    )

    args = parser.parse_args()

    client = VADClient(
        config_path=args.config,
        server_url=args.server,
        auth_token=args.auth_token
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nExited")


if __name__ == "__main__":
    main()
