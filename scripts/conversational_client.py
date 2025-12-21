#!/usr/bin/env python3
"""
Conversational Client - Robust cancellation and interrupt handling.

Features:
- asyncio.Event for thread-safe cancellation signaling
- Generation counters in all components to filter stale data
- Timeout-based network calls to break blocking
- Atomic interrupt handling with queue purging

Uses servers for: Whisper (transcription), LLM, TTS
Local: VAD, audio playback
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
import time
from queue import Queue
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio import AudioPlayer
from src.services import LLMClient, TTSClient

# VAD and Segmentor (local processing)
from src.vad_ten import TenVAD
from src.segmentor import Segmentor


def timestamp():
    """Get current timestamp in ms for logging."""
    return f"[{time.time()*1000:.0f}]"


class State(Enum):
    """Client state machine states."""
    IDLE = "idle"
    PROCESSING = "processing"


@dataclass
class Message:
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str


class ConversationalClient:
    """
    Conversational client with robust interrupt handling.

    Key features:
    - Global generation counter for invalidating stale data
    - asyncio.Event based cancellation across all components
    - Timeout-based network calls
    - Atomic interrupt with queue purging
    """

    def __init__(self, config_path: str):
        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Audio config
        self.sample_rate = self.cfg["audio"]["sample_rate"]
        self.frame_ms = self.cfg["audio"]["frame_ms"]
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        # Initialize VAD and Segmentor (local, no cancellation needed)
        print("Loading VAD model...")
        self.vad = TenVAD(self.cfg["vad"]["model_path"])
        self.segmentor = Segmentor(self.cfg)
        print("VAD initialized")

        # Audio queue (thread-safe, for mic input)
        self.audio_queue = Queue(maxsize=500)

        # Initialize v2 clients with robust cancellation
        self.llm_client = LLMClient(
            self.cfg["llm"]["host"],
            self.cfg["llm"]["port"],
            self.cfg["llm"]["model"],
            self.cfg["llm"]["temperature"],
            self.cfg["llm"]["max_tokens"]
        )

        # TTS client (server only in v2)
        recv_timeout = self.cfg["tts"].get("recv_timeout_ms", 100) / 1000.0
        self.tts_client = TTSClient(
            url=self.cfg["tts"]["url"],
            auth_token=self.cfg["tts"].get("auth_token"),
            recv_timeout=recv_timeout
        )
        print(f"Using TTS server ({self.cfg['tts']['url']})")

        # Audio player with generation tracking
        self.audio_player = AudioPlayer(
            self.cfg["playback"]["fade_out_duration_ms"],
            self.cfg["playback"]["output_device"]
        )

        # State machine
        self.state = State.IDLE
        self.conversation_history: List[Message] = []
        self.system_prompt = self.cfg["llm"]["system_prompt"]
        self.max_history = self.cfg["conversation"]["max_history"]

        # Current message (accumulates during interrupts)
        self.current_message: Optional[str] = None

        # Speech detection tracking
        self.was_in_speech = False

        # Track sentences in current response
        self.current_assistant_sentences = []

        # Global generation counter for atomic invalidation
        self.generation = 0

        # Stats
        self.segments_received = 0
        self.mic_frame_count = 0
        self.loop_iteration_count = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice - runs in audio thread."""
        self.mic_frame_count += 1
        if self.mic_frame_count % 500 == 0:
            print(f"[Mic] {self.mic_frame_count} frames received")

        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Convert float32 to int16
        pcm16 = (indata[:, 0] * 32767).astype(np.int16)

        try:
            self.audio_queue.put_nowait(pcm16)
        except:
            pass  # Drop frame if queue full

    def encode_segment(self, pcm_bytes: bytes, start_iso: str, end_iso: str) -> bytes:
        """Encode segment for transmission to Whisper server."""
        header = {
            "start_utc": start_iso,
            "end_utc": end_iso
        }

        language = self.cfg["whisper_server"].get("language")
        if language:
            header["language"] = language

        header_bytes = json.dumps(header).encode('utf-8')
        return struct.pack('>I', len(header_bytes)) + header_bytes + pcm_bytes

    async def receive_transcriptions(self, ws):
        """Receive transcriptions from Whisper server."""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    if data.get("type") == "asr_segment":
                        text = data.get("text", "").strip()
                        if text:
                            await self.handle_transcription(text)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"Error handling transcription: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error receiving transcriptions: {e}", file=sys.stderr)

    async def handle_transcription(self, text: str):
        """Handle incoming transcription based on current state."""
        self.segments_received += 1
        print(f"\n{timestamp()} [User said] {text}")

        if self.state == State.IDLE:
            self.current_message = text
            await self.process_user_message()

        elif self.state == State.PROCESSING:
            print(f"{timestamp()} [Interrupt] Transcription during processing")

            if self.current_message:
                self.current_message = f"{self.current_message}. {text}"
            else:
                self.current_message = text

            # Wait for user to finish speaking
            await asyncio.sleep(self.cfg["conversation"]["interrupt_timeout_s"])
            await self.process_user_message()

    async def process_user_message(self):
        """Process current_message through LLM."""
        if not self.current_message:
            return

        if self.state != State.PROCESSING:
            self.state = State.PROCESSING
            print("[State] → PROCESSING")

        # Reset for new response
        self.current_assistant_sentences = []
        self.audio_player.reset_tracking()

        # Reset cancellation state for new turn
        self.llm_client.reset_for_new_request()
        self.tts_client.reset_for_new_request()
        self.audio_player.reset_for_new_turn()

        print(f"{timestamp()} [Processing] {self.current_message}")

        # Add to history
        self.conversation_history.append(Message(role="user", content=self.current_message))

        # Trim history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

        # Prepare messages for LLM
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        print(f"{timestamp()} [LLM] Request enqueued")
        self.llm_client.enqueue(messages)

    async def _handle_interrupt(self):
        """
        Atomic interrupt handler.
        Called when speech detected during PROCESSING state.
        """
        print(f"{timestamp()} [Interrupt] Speech detected - aborting!")

        # 1. Capture state BEFORE abort
        played_count = self.audio_player.chunks_completed

        # 2. Bump global generation (invalidates all in-flight)
        self.generation += 1

        # 3. Signal all components to abort
        self.llm_client.abort()
        self.tts_client.abort()
        self.audio_player.abort()

        # 4. Brief pause for queues to drain
        await asyncio.sleep(0.05)

        # 5. Force purge any stragglers
        self.audio_player.clear_queue()

        # 6. Update history with only played content
        if played_count > 0 and self.current_assistant_sentences:
            played = self.current_assistant_sentences[:played_count]
            self.conversation_history.append(Message(
                role="assistant",
                content=" ".join(played)
            ))
            print(f"[Interrupt] Kept {played_count} fully-played sentence(s)")

        # 7. Reset tracking
        self.current_assistant_sentences = []
        self.audio_player.reset_tracking()

    async def _llm_to_tts_forwarder(self):
        """Forward LLM sentences to TTS."""
        try:
            while True:
                for sentence in self.llm_client.get_ready_sentences():
                    print(f"{timestamp()} [LLM→TTS] {sentence[:60]}...")
                    self.tts_client.enqueue(sentence)
                    self.current_assistant_sentences.append(sentence)

                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass

    async def _tts_to_audio_forwarder(self):
        """Forward TTS audio to AudioPlayer."""
        try:
            while True:
                for wav_bytes in self.tts_client.get_ready_audio():
                    print(f"{timestamp()} [TTS→Audio] {len(wav_bytes)} bytes")
                    self.audio_player.enqueue(wav_bytes)

                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass

    async def _state_monitor(self):
        """Monitor service states and transition to IDLE when done."""
        try:
            while True:
                if self.state == State.PROCESSING:
                    all_idle = (
                        not self.llm_client.is_processing() and
                        not self.tts_client.is_processing() and
                        not self.audio_player.is_playing
                    )

                    if all_idle:
                        # Commit all sentences to history
                        if self.current_assistant_sentences:
                            self.conversation_history.append(Message(
                                role="assistant",
                                content=" ".join(self.current_assistant_sentences)
                            ))
                            self.current_assistant_sentences = []

                        self.audio_player.reset_tracking()
                        print(f"{timestamp()} [State] → IDLE")
                        self.state = State.IDLE
                        self.current_message = None

                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    async def run(self):
        """Main loop."""
        server_url = self.cfg["whisper_server"]["url"]
        auth_token = self.cfg["whisper_server"].get("auth_token")

        print(f"Connecting to Whisper server at {server_url}...")

        # Start audio player
        self.audio_player.start()

        # Initialize clients
        async with self.llm_client, self.tts_client:
            self.llm_client.start()
            self.tts_client.start()

            async with websockets.connect(server_url, max_size=10*1024*1024) as ws:
                print("Connected to Whisper server")

                if auth_token:
                    await ws.send(auth_token)

                # Start background tasks
                tasks = [
                    asyncio.create_task(self.receive_transcriptions(ws)),
                    asyncio.create_task(self._llm_to_tts_forwarder()),
                    asyncio.create_task(self._tts_to_audio_forwarder()),
                    asyncio.create_task(self._state_monitor()),
                ]

                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    blocksize=self.frame_samples,
                    callback=self.audio_callback
                ):
                    print(f"Listening on microphone ({self.sample_rate}Hz)")
                    print("Speak to chat with the AI assistant...")
                    print("Press Ctrl+C to stop\n")

                    try:
                        while True:
                            await asyncio.sleep(0.01)
                            self.loop_iteration_count += 1

                            # Drain audio queue
                            frames = []
                            while not self.audio_queue.empty():
                                try:
                                    frames.append(self.audio_queue.get_nowait())
                                except:
                                    break

                            # Heartbeat with VAD diagnostics
                            if self.loop_iteration_count % 1000 == 0:
                                print(f"[Heartbeat] loop={self.loop_iteration_count}, "
                                      f"state={self.state.value}, gen={self.generation}, "
                                      f"in_speech={self.segmentor.in_speech}, "
                                      f"prob_ema={self.segmentor.prob_ema:.3f}")

                            # Process VAD
                            for frame in frames:
                                prob = self.vad.prob_speech(frame)
                                result = self.segmentor.update(frame, prob)

                                # Interrupt detection (speech edge)
                                if self.segmentor.in_speech and not self.was_in_speech:
                                    if self.state == State.PROCESSING:
                                        await self._handle_interrupt()

                                # Log speech transitions
                                if self.segmentor.in_speech != self.was_in_speech:
                                    print(f"[Speech] {self.was_in_speech} → {self.segmentor.in_speech}")

                                self.was_in_speech = self.segmentor.in_speech

                                if result:
                                    pcm_bytes, start_iso, end_iso = result
                                    duration_s = len(pcm_bytes) / 2 / self.sample_rate
                                    print(f"\n{timestamp()} [VAD] Segment ({duration_s:.2f}s)")

                                    message = self.encode_segment(pcm_bytes, start_iso, end_iso)
                                    try:
                                        await asyncio.wait_for(ws.send(message), timeout=5.0)
                                    except asyncio.TimeoutError:
                                        print("Timeout sending to Whisper", file=sys.stderr)
                                    except Exception as e:
                                        print(f"Error sending segment: {e}", file=sys.stderr)

                    except KeyboardInterrupt:
                        print("\nStopping...")
                        for task in tasks:
                            task.cancel()


def main():
    parser = argparse.ArgumentParser(
        description="Conversational client with robust cancellation"
    )
    parser.add_argument(
        "--config",
        default="config.conversational_client.yaml",
        help="Config file path"
    )

    args = parser.parse_args()
    client = ConversationalClient(config_path=args.config)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nExited")


if __name__ == "__main__":
    main()
