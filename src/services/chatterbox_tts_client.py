"""Client for Chatterbox-Turbo TTS (Text-to-Speech) from Resemble AI"""
import sys
import asyncio
import io
from typing import Optional, List

import torch
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS


class ChatterboxTTSClient:
    """Queue-based Chatterbox-Turbo TTS client with background synthesis"""

    def __init__(self, device: str = "auto"):
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Chatterbox-Turbo model on {self.device}...")
        self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        print("Chatterbox-Turbo model loaded")

        # Queue for text to synthesize (will be created when loop starts)
        self.text_queue: Optional[asyncio.Queue] = None

        # List of completed audio ready for pickup
        self.ready_audio: List[bytes] = []

        # Control flags
        self.should_stop = False
        self._synthesis_task: Optional[asyncio.Task] = None
        self._started = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def start(self) -> None:
        """Start the synthesis loop (call once inside async context)"""
        if not self._started:
            self.text_queue = asyncio.Queue()
            self._synthesis_task = asyncio.create_task(self._synthesis_loop())
            self._started = True

    def enqueue(self, text: str) -> None:
        """Add text to synthesis queue (non-blocking)"""
        if not self._started:
            raise RuntimeError("ChatterboxTTSClient not started. Call start() first in async context.")
        self.text_queue.put_nowait(text)

    def get_ready_audio(self) -> List[bytes]:
        """
        Get all completed audio and clear the ready list (non-blocking, synchronous).
        Returns list of WAV bytes.
        """
        if not self.ready_audio:
            return []

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
                wav_bytes = await self._synthesize(text)

                # Only add to ready list if not aborted
                if not self.should_stop and wav_bytes:
                    self.ready_audio.append(wav_bytes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Synthesis loop error: {e}", file=sys.stderr)

    async def _synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize a single text to WAV bytes using Chatterbox-Turbo"""
        try:
            # Run the blocking model.generate() in a thread pool
            wav = await asyncio.to_thread(
                self.model.generate,
                text
            )

            # Check if aborted
            if self.should_stop:
                return None

            # Convert tensor to WAV bytes
            buffer = io.BytesIO()
            # wav is [1, samples] tensor, convert to numpy [samples]
            audio_np = wav.squeeze(0).cpu().numpy()
            sf.write(buffer, audio_np, self.model.sr, format="WAV")
            buffer.seek(0)
            return buffer.read()

        except asyncio.CancelledError:
            return None
        except Exception as e:
            import traceback
            print(f"Chatterbox TTS error: {e}", file=sys.stderr)
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
