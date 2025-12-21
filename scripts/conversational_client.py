#!/usr/bin/env python3
"""
Conversational VAD Client with LLM + TTS
Captures speech, transcribes, generates LLM response, and plays back audio.
Handles interruptions intelligently.
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
import aiohttp
import re
from queue import Queue
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, AsyncIterator
from dataclasses import dataclass
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio import AudioPlayer
from src.services import TTSClient, ElevenLabsTTSClient, LLMClient, ChatterboxTTSClient, WebSocketTTSClient


def timestamp():
    """Get current timestamp in ms for logging"""
    return f"[{time.time()*1000:.0f}]"


class State(Enum):
    """Client state machine states"""
    IDLE = "idle"
    PROCESSING = "processing"


@dataclass
class Message:
    """Chat message"""
    role: str  # "user" or "assistant"
    content: str


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for streaming TTS.
    Handles common abbreviations to avoid false splits.
    """
    # Replace common abbreviations temporarily
    text = text.replace("Dr.", "Dr<DOT>")
    text = text.replace("Mr.", "Mr<DOT>")
    text = text.replace("Mrs.", "Mrs<DOT>")
    text = text.replace("Ms.", "Ms<DOT>")
    text = text.replace("U.S.", "U<DOT>S<DOT>")
    text = text.replace("U.K.", "U<DOT>K<DOT>")
    text = text.replace("etc.", "etc<DOT>")
    text = text.replace("vs.", "vs<DOT>")
    text = text.replace("e.g.", "e<DOT>g<DOT>")
    text = text.replace("i.e.", "i<DOT>e<DOT>")

    # Split on sentence boundaries (.!?) followed by space or end
    sentences = re.split(r'([.!?]+)(?:\s+|$)', text)

    # Recombine sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""

        if sentence:
            # Restore abbreviations
            sentence = sentence.replace("<DOT>", ".")
            combined = (sentence + punctuation).strip()
            if combined:
                result.append(combined)

    # Handle any remaining text without punctuation
    if sentences and sentences[-1].strip():
        remaining = sentences[-1].strip().replace("<DOT>", ".")
        if remaining:
            result.append(remaining)

    return result if result else [text]  # Fallback to original if no splits


class ConversationalClient:
    """
    Main conversational client with VAD, transcription, LLM, TTS, and audio playback.
    Handles interruptions intelligently.
    """

    def __init__(self, config_path: str):
        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Audio config
        self.sample_rate = self.cfg["audio"]["sample_rate"]
        self.frame_ms = self.cfg["audio"]["frame_ms"]
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        # Import VAD and Segmentor
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

        # Clients
        self.llm_client = LLMClient(
            self.cfg["llm"]["host"],
            self.cfg["llm"]["port"],
            self.cfg["llm"]["model"],
            self.cfg["llm"]["temperature"],
            self.cfg["llm"]["max_tokens"]
        )

        # Initialize TTS client based on provider
        tts_provider = self.cfg["tts"].get("provider", "local")
        if tts_provider == "elevenlabs":
            elevenlabs_cfg = self.cfg["tts"]["elevenlabs"]
            self.tts_client = ElevenLabsTTSClient(
                elevenlabs_cfg["voice_id"],
                elevenlabs_cfg["model_id"],
                elevenlabs_cfg["output_format"]
            )
            print(f"Using ElevenLabs TTS (voice: {elevenlabs_cfg['voice_id']})")
        elif tts_provider == "chatterbox":
            chatterbox_cfg = self.cfg["tts"]["chatterbox"]
            self.tts_client = ChatterboxTTSClient(
                device=chatterbox_cfg.get("device", "auto")
            )
            print(f"Using Chatterbox-Turbo TTS (device: {chatterbox_cfg.get('device', 'auto')})")
        elif tts_provider == "server":
            server_cfg = self.cfg["tts"]["server"]
            self.tts_client = WebSocketTTSClient(
                url=server_cfg["url"],
                auth_token=server_cfg.get("auth_token")
            )
            print(f"Using TTS server ({server_cfg['url']})")
        else:
            local_cfg = self.cfg["tts"]["local"]
            self.tts_client = TTSClient(
                local_cfg["host"],
                local_cfg["port"],
                local_cfg["endpoint"]
            )
            print(f"Using local TTS ({local_cfg['host']}:{local_cfg['port']})")
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

        # Speech detection tracking (for immediate interrupts)
        self.was_in_speech = False

        # Track sentences in current assistant response (for proper history management)
        self.current_assistant_sentences = []

        # Stats
        self.segments_received = 0

        # Diagnostic counters for debugging hangs
        self.mic_frame_count = 0
        self.loop_iteration_count = 0
        self.no_frame_iterations = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice - runs in audio thread."""
        # Track frame reception for diagnostics
        self.mic_frame_count += 1
        if self.mic_frame_count % 500 == 0:  # Every ~15s at 30ms frames
            print(f"[Mic] {self.mic_frame_count} frames received (queue: {self.audio_queue.qsize()})")

        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        pcm16 = (indata[:, 0] * 32767).astype(np.int16)

        # Put in async queue (non-blocking)
        try:
            self.audio_queue.put_nowait(pcm16)
        except:
            pass  # Drop frame if queue full

    def encode_segment(self, pcm_bytes: bytes, start_iso: str, end_iso: str) -> bytes:
        """Encode segment for transmission to Whisper server"""
        header = {
            "start_utc": start_iso,
            "end_utc": end_iso
        }

        # Add language hint if specified
        language = self.cfg["whisper_server"].get("language")
        if language:
            header["language"] = language

        header_bytes = json.dumps(header).encode('utf-8')
        header_len = len(header_bytes)

        return struct.pack('>I', header_len) + header_bytes + pcm_bytes

    async def receive_transcriptions(self, ws):
        """Receive transcriptions from Whisper server"""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    if data.get("type") == "asr_segment":
                        text = data.get("text", "").strip()
                        if text:
                            await self.handle_transcription(text)
                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}")
                except Exception as e:
                    print(f"Error handling transcription: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"Error receiving transcriptions: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    async def handle_transcription(self, text: str):
        """Handle incoming transcription based on current state"""
        self.segments_received += 1
        print(f"\n{timestamp()} [User said] {text}")

        if self.state == State.IDLE:
            # Start new conversation turn
            self.current_message = text
            await self.process_user_message()

        elif self.state == State.PROCESSING:
            # User interrupted - abort already happened in main loop
            # Just accumulate the message
            print(f"{timestamp()} [Interrupt] Transcription received during processing")

            # Concatenate with current message
            if self.current_message:
                self.current_message = f"{self.current_message}. {text}"
            else:
                self.current_message = text

            print(f"[Interrupt] Accumulated message: {self.current_message}")

            # Wait for user to finish speaking (timeout mechanism)
            # If another transcription comes, it will accumulate further
            await asyncio.sleep(self.cfg["conversation"]["interrupt_timeout_s"])

            # After timeout, process the accumulated message
            await self.process_user_message()

    async def process_user_message(self):
        """Process current_message: enqueue to LLM (main loop handles rest)"""
        if not self.current_message:
            print("[Warning] process_user_message called with no current_message")
            return

        # Ensure we're in PROCESSING state
        if self.state != State.PROCESSING:
            self.state = State.PROCESSING
            print("[State] → PROCESSING")

        # Reset for new response
        self.current_assistant_sentences = []
        self.audio_player.reset_tracking()

        print(f"{timestamp()} [Processing] {self.current_message}")

        # Add to history
        self.conversation_history.append(Message(role="user", content=self.current_message))

        # Trim history if too long
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

        # Prepare messages for LLM
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Enqueue to LLM (non-blocking)
        print(f"{timestamp()} [LLM] Request enqueued")
        self.llm_client.enqueue(messages)

    async def _llm_to_tts_forwarder(self):
        """Background task: forward LLM sentences to TTS"""
        try:
            while True:
                for sentence in self.llm_client.get_ready_sentences():
                    print(f"{timestamp()} [LLM→TTS] {sentence[:60]}...")
                    self.tts_client.enqueue(sentence)

                    # Track sentence (will be added to history only when played)
                    self.current_assistant_sentences.append(sentence)

                await asyncio.sleep(0.001)  # Yield control
        except asyncio.CancelledError:
            pass

    async def _tts_to_audio_forwarder(self):
        """Background task: forward TTS audio to AudioPlayer"""
        try:
            while True:
                for wav_bytes in self.tts_client.get_ready_audio():
                    print(f"{timestamp()} [TTS→Audio] Audio chunk ready ({len(wav_bytes)} bytes)")
                    self.audio_player.enqueue(wav_bytes)

                await asyncio.sleep(0.001)  # Yield control
        except asyncio.CancelledError:
            pass

    async def _state_monitor(self):
        """Background task: monitor service states and transition to IDLE when done"""
        try:
            while True:
                if self.state == State.PROCESSING:
                    all_idle = (
                        not self.llm_client.is_processing() and
                        not self.tts_client.is_processing() and
                        not self.audio_player.is_playing
                    )
                    if all_idle:
                        # Commit all sentences to history (response completed normally)
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

                await asyncio.sleep(0.01)  # Check every 10ms
        except asyncio.CancelledError:
            pass

    async def run(self):
        """Main loop: connect to Whisper server and process audio"""
        server_url = self.cfg["whisper_server"]["url"]
        auth_token = self.cfg["whisper_server"].get("auth_token")

        print(f"Connecting to Whisper server at {server_url}...")

        # Start audio player, TTS client, and LLM client
        self.audio_player.start()

        # Initialize clients
        async with self.llm_client, self.tts_client:
            self.llm_client.start()
            self.tts_client.start()
            async with websockets.connect(server_url, max_size=10*1024*1024) as ws:
                print(f"Connected to Whisper server")

                # Send auth token if required
                if auth_token:
                    await ws.send(auth_token)
                    print("Sent authentication token")

                # Start background tasks
                receive_task = asyncio.create_task(self.receive_transcriptions(ws))
                llm_tts_task = asyncio.create_task(self._llm_to_tts_forwarder())
                tts_audio_task = asyncio.create_task(self._tts_to_audio_forwarder())
                state_monitor_task = asyncio.create_task(self._state_monitor())

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
                        # Main loop: focused on VAD processing only
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

                            # Heartbeat every 10s to prove main loop is running
                            if self.loop_iteration_count % 1000 == 0:
                                print(f"[Heartbeat] loop={self.loop_iteration_count}, mic_frames={self.mic_frame_count}, "
                                      f"state={self.state.value}, playing={self.audio_player.is_playing}")

                            # Diagnostic: warn if no frames while audio is playing
                            if not frames:
                                if self.audio_player.is_playing:
                                    self.no_frame_iterations += 1
                                    if self.no_frame_iterations == 50:  # 500ms with no frames
                                        print(f"[Warning] No mic frames for 500ms while audio playing!")
                                    elif self.no_frame_iterations % 100 == 0:  # Every 1s after that
                                        print(f"[Warning] No mic frames for {self.no_frame_iterations * 10}ms!")
                            else:
                                if self.no_frame_iterations >= 50:
                                    print(f"[Mic] Frames resumed after {self.no_frame_iterations * 10}ms pause")
                                self.no_frame_iterations = 0

                            # Process VAD on all frames
                            for frame in frames:
                                prob = self.vad.prob_speech(frame)
                                result = self.segmentor.update(frame, prob)

                                # Check if speech just started (immediate interrupt detection)
                                if self.segmentor.in_speech and not self.was_in_speech:
                                    # Speech START detected!
                                    if self.state == State.PROCESSING:
                                        print(f"{timestamp()} [Interrupt] Speech detected - aborting immediately!")

                                        # Get count of fully played chunks before aborting
                                        played_count = self.audio_player.chunks_completed

                                        # Abort all services
                                        self.llm_client.abort()
                                        self.tts_client.abort()
                                        self.audio_player.abort()

                                        # Keep only sentences that were fully played
                                        if played_count > 0 and self.current_assistant_sentences:
                                            played_sentences = self.current_assistant_sentences[:played_count]
                                            self.conversation_history.append(Message(
                                                role="assistant",
                                                content=" ".join(played_sentences)
                                            ))
                                            print(f"[Interrupt] Kept {played_count} fully-played sentence(s) in history")

                                        # Clear current response tracking
                                        self.current_assistant_sentences = []
                                        self.audio_player.reset_tracking()

                                # Log speech state transitions for debugging
                                if self.segmentor.in_speech != self.was_in_speech:
                                    print(f"[Speech] {self.was_in_speech} → {self.segmentor.in_speech} (state={self.state.value})")

                                self.was_in_speech = self.segmentor.in_speech

                                if result:
                                    # Segment complete - send to Whisper server
                                    pcm_bytes, start_iso, end_iso = result
                                    duration_s = len(pcm_bytes) / 2 / self.sample_rate

                                    print(f"\n{timestamp()} [VAD] Speech segment detected ({duration_s:.2f}s)")

                                    message = self.encode_segment(pcm_bytes, start_iso, end_iso)

                                    try:
                                        await asyncio.wait_for(ws.send(message), timeout=5.0)
                                    except asyncio.TimeoutError:
                                        print(f"⚠ Timeout sending to Whisper server", file=sys.stderr)
                                    except Exception as e:
                                        print(f"⚠ Error sending segment: {e}", file=sys.stderr)

                    except KeyboardInterrupt:
                        print("\nStopping...")
                        receive_task.cancel()
                        llm_tts_task.cancel()
                        tts_audio_task.cancel()
                        state_monitor_task.cancel()


def main():
    parser = argparse.ArgumentParser(
        description="Conversational VAD client with LLM and TTS"
    )
    parser.add_argument(
        "--config",
        default="config.conversational_client.yaml",
        help="Config file path (default: config.conversational_client.yaml)"
    )

    args = parser.parse_args()

    client = ConversationalClient(config_path=args.config)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nExited")


if __name__ == "__main__":
    main()
