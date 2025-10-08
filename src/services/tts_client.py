"""Client for TTS (Text-to-Speech) service"""
import sys
import asyncio
import aiohttp
from typing import Optional, List


class TTSClient:
    """Queue-based TTS client with background synthesis"""

    def __init__(self, host: str, port: int, endpoint: str):
        self.url = f"http://{host}:{port}{endpoint}"
        self.session: Optional[aiohttp.ClientSession] = None

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
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def start(self) -> None:
        """Start the synthesis loop (call once inside async context)"""
        if not self._started:
            self.text_queue = asyncio.Queue()
            self._synthesis_task = asyncio.create_task(self._synthesis_loop())
            self._started = True

    def enqueue(self, text: str) -> None:
        """Add text to synthesis queue (non-blocking)"""
        if not self._started:
            raise RuntimeError("TTSClient not started. Call start() first in async context.")
        self.text_queue.put_nowait(text)

    def get_ready_audio(self) -> List[bytes]:
        """
        Get all completed audio and clear the ready list (non-blocking, synchronous).
        Returns list of WAV bytes.
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
                wav_bytes = await self._synthesize(text)

                # Only add to ready list if not aborted
                if not self.should_stop and wav_bytes:
                    self.ready_audio.append(wav_bytes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Synthesis loop error: {e}", file=sys.stderr)

    async def _synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize a single text to WAV bytes"""
        if not self.session:
            raise RuntimeError("TTSClient not initialized. Use 'async with' context manager.")

        payload = {"text": text}

        try:
            async with self.session.post(
                self.url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                return await response.read()
        except asyncio.CancelledError:
            # Synthesis was aborted
            return None
        except Exception as e:
            import traceback
            print(f"TTS API error: {e}", file=sys.stderr)
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
