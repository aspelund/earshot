"""Client for ElevenLabs TTS (Text-to-Speech) service"""
import sys
import asyncio
import os
from typing import Optional, List
from datetime import datetime
from elevenlabs.client import ElevenLabs


class ElevenLabsTTSClient:
    """Queue-based ElevenLabs TTS client with background synthesis"""

    def __init__(self, voice_id: str, model_id: str, output_format: str, usage_log_path: str = "elevenlabs_usage.log"):
        # Load API key from environment
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")

        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.usage_log_path = usage_log_path

        # Queue for text to synthesize (will be created when loop starts)
        self.text_queue: Optional[asyncio.Queue] = None

        # List of completed audio ready for pickup
        self.ready_audio: List[bytes] = []
        self._ready_lock = asyncio.Lock()

        # Control flags
        self.should_stop = False
        self._synthesis_task: Optional[asyncio.Task] = None
        self._started = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _log_usage(self, text: str):
        """Log character usage to file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            char_count = len(text)
            log_entry = f"{timestamp} - {char_count} chars - \"{text[:100]}{'...' if len(text) > 100 else ''}\"\n"

            with open(self.usage_log_path, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Warning: Failed to log usage: {e}", file=sys.stderr)

    def start(self) -> None:
        """Start the synthesis loop (call once inside async context)"""
        if not self._started:
            self.text_queue = asyncio.Queue()
            self._synthesis_task = asyncio.create_task(self._synthesis_loop())
            self._started = True

    def enqueue(self, text: str) -> None:
        """Add text to synthesis queue (non-blocking)"""
        if not self._started:
            raise RuntimeError("ElevenLabsTTSClient not started. Call start() first in async context.")
        self.text_queue.put_nowait(text)

    def get_ready_audio(self) -> List[bytes]:
        """
        Get all completed audio and clear the ready list (non-blocking, synchronous).
        Returns list of audio bytes (in configured format, e.g., MP3).
        """
        if not self.ready_audio:
            return []

        # Get all ready audio and clear the list
        result = self.ready_audio.copy()
        self.ready_audio.clear()
        return result

    def is_processing(self) -> bool:
        """Returns True if has queued text or ready audio"""
        has_queued_text = self.text_queue and not self.text_queue.empty()
        has_ready_audio = len(self.ready_audio) > 0
        return has_queued_text or has_ready_audio

    def abort(self) -> None:
        """Clear text queue and ready audio (non-blocking)"""
        # Clear text queue
        if self.text_queue:
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Clear ready audio
        self.ready_audio.clear()

        # Set stop flag to abort current synthesis
        self.should_stop = True

    async def _synthesis_loop(self):
        """Background task that continuously synthesizes text from queue"""
        while True:
            try:
                # Wait for next text item
                text = await self.text_queue.get()

                # Reset stop flag for new synthesis
                self.should_stop = False

                # Synthesize
                audio_bytes = await self._synthesize(text)

                # Only add to ready list if not aborted
                if not self.should_stop and audio_bytes:
                    self.ready_audio.append(audio_bytes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Synthesis loop error: {e}", file=sys.stderr)

    async def _synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize a single text to audio bytes using ElevenLabs API"""
        try:
            # Run the blocking ElevenLabs API call in a thread pool
            audio_iterator = await asyncio.to_thread(
                self.client.text_to_speech.convert,
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=self.output_format
            )

            # Check if aborted before reading
            if self.should_stop:
                return None

            # Collect all audio chunks from the iterator
            audio_chunks = []
            for chunk in audio_iterator:
                if self.should_stop:
                    return None
                audio_chunks.append(chunk)

            # Combine all chunks into a single bytes object
            audio_bytes = b''.join(audio_chunks)

            # Log usage
            self._log_usage(text)

            return audio_bytes

        except asyncio.CancelledError:
            # Synthesis was aborted
            return None
        except Exception as e:
            import traceback
            print(f"ElevenLabs TTS API error: {e}", file=sys.stderr)
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
            return None

    async def stop(self):
        """Stop the synthesis loop (cleanup)"""
        if self._synthesis_task:
            self._synthesis_task.cancel()
            try:
                await self._synthesis_task
            except asyncio.CancelledError:
                pass
