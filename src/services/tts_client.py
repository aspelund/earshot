"""
TTS WebSocket client with robust cancellation support.

Improvements over original:
- asyncio.Event for thread-safe cancellation signaling
- Timeout-based recv() to break blocking and check cancellation
- Generation counter to filter stale audio responses
- Double-check generation before AND after network calls
"""

import sys
import asyncio
import json
import uuid
from typing import Optional, List

import websockets


class TTSClient:
    """TTS WebSocket client with asyncio.Event cancellation and timeout-based recv."""

    def __init__(self, url: str, auth_token: str = None, recv_timeout: float = 0.1):
        self.url = url
        self.auth_token = auth_token
        self.recv_timeout = recv_timeout  # Timeout for recv() in seconds

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

        # Queue for text to synthesize
        self.text_queue: Optional[asyncio.Queue] = None

        # Completed audio ready for pickup
        self.ready_audio: List[bytes] = []

        # Track pending requests
        self.pending_requests: dict = {}  # request_id -> text

        # Cancellation via asyncio.Event
        self.cancel_event = asyncio.Event()

        # Generation counter for filtering stale responses
        self.generation = 0
        self.active_generation = 0

        # Background tasks
        self._synthesis_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._started = False
        self._connected = False

    async def __aenter__(self):
        """Connect to TTS server."""
        try:
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10
            )
            self._connected = True

            # Authenticate if needed
            if self.auth_token:
                await self.websocket.send(self.auth_token)

            print(f"Connected to TTS server at {self.url}")
        except Exception as e:
            print(f"Failed to connect to TTS server: {e}", file=sys.stderr)
            raise

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Disconnect from TTS server."""
        if self.websocket:
            await self.websocket.close()
            self._connected = False

    def start(self) -> None:
        """Start background tasks (call once inside async context)."""
        if not self._started:
            self.text_queue = asyncio.Queue()
            self._synthesis_task = asyncio.create_task(self._synthesis_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._started = True

    def enqueue(self, text: str) -> None:
        """Add text to synthesis queue (non-blocking)."""
        if not self._started:
            raise RuntimeError("TTSClient not started. Call start() first.")

        # Only enqueue if not cancelled
        if not self.cancel_event.is_set():
            self.text_queue.put_nowait(text)

    def get_ready_audio(self) -> List[bytes]:
        """
        Get all completed audio and clear the list (non-blocking).
        Returns list of WAV bytes.
        """
        if not self.ready_audio:
            return []

        result = self.ready_audio.copy()
        self.ready_audio.clear()
        return result

    def is_processing(self) -> bool:
        """Returns True if has queued text, pending requests, or ready audio."""
        has_queued = self.text_queue and not self.text_queue.empty()
        has_pending = len(self.pending_requests) > 0
        has_ready = len(self.ready_audio) > 0
        return has_queued or has_pending or has_ready

    def abort(self) -> None:
        """
        Atomic abort: increment generation, signal cancellation, purge all state.
        """
        # 1. Increment generation (invalidates all in-flight responses)
        self.generation += 1

        # 2. Signal cancellation (wakes sleeping recv)
        self.cancel_event.set()

        # 3. Purge all state
        self._purge_queues()

        # 4. Send cancel messages to server for pending requests
        if self._connected and self.websocket:
            for request_id in list(self.pending_requests.keys()):
                asyncio.create_task(self._send_cancel(request_id))

    def _purge_queues(self) -> None:
        """Clear all queues and pending state."""
        # Clear text queue
        if self.text_queue:
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Clear pending requests
        self.pending_requests.clear()

        # Clear ready audio
        self.ready_audio.clear()

    def reset_for_new_request(self) -> None:
        """Reset cancellation state for new request."""
        self.cancel_event.clear()
        self.active_generation = self.generation

    async def _send_cancel(self, request_id: str):
        """Send cancel message to server."""
        try:
            await self.websocket.send(json.dumps({
                "type": "cancel",
                "request_id": request_id
            }))
        except Exception:
            pass

    async def _synthesis_loop(self):
        """Background task that sends text to server."""
        while True:
            try:
                # Wait for text with timeout to check cancellation
                try:
                    text = await asyncio.wait_for(
                        self.text_queue.get(),
                        timeout=self.recv_timeout
                    )
                except asyncio.TimeoutError:
                    continue

                # Check cancellation before sending
                if self.cancel_event.is_set():
                    continue

                # Capture generation for this request
                self.active_generation = self.generation
                gen_at_request = self.active_generation

                # Generate request ID
                request_id = str(uuid.uuid4())

                # Track pending request with generation
                self.pending_requests[request_id] = {
                    "text": text,
                    "generation": gen_at_request
                }

                # Send to server
                if self._connected and self.websocket:
                    await self.websocket.send(json.dumps({
                        "type": "synthesize",
                        "text": text,
                        "request_id": request_id
                    }))

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Synthesis loop error: {e}", file=sys.stderr)

    async def _receive_loop(self):
        """
        Background task that receives audio from server.
        Uses timeout-based recv to periodically check cancellation.
        """
        while True:
            try:
                if not self._connected or not self.websocket:
                    await asyncio.sleep(0.1)
                    continue

                # Timeout-based recv - allows checking cancellation periodically
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.recv_timeout
                    )
                except asyncio.TimeoutError:
                    # No message, loop back and check cancellation
                    continue

                # Check cancellation AFTER receiving
                if self.cancel_event.is_set():
                    # Discard message, we're cancelled
                    continue

                # Check generation AFTER receiving
                if self.active_generation != self.generation:
                    # Stale response, discard
                    continue

                if isinstance(message, bytes):
                    # Binary audio data - add to ready queue
                    self.ready_audio.append(message)
                    # Clear oldest pending request (server processes in order)
                    if self.pending_requests:
                        oldest_key = next(iter(self.pending_requests))
                        self.pending_requests.pop(oldest_key, None)
                else:
                    # JSON response (error, cancelled, etc.)
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        request_id = data.get("request_id")

                        # Remove from pending
                        self.pending_requests.pop(request_id, None)

                        if msg_type == "error":
                            print(f"TTS server error: {data.get('error')}", file=sys.stderr)
                        elif msg_type == "cancelled":
                            pass  # Expected when we abort

                    except json.JSONDecodeError:
                        pass

            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosed:
                print("TTS server connection closed", file=sys.stderr)
                self._connected = False
                # Clear pending state so is_processing() returns False
                self.pending_requests.clear()
                self.ready_audio.clear()
                break
            except Exception as e:
                print(f"Receive loop error: {e}", file=sys.stderr)

    async def stop(self):
        """Stop background tasks (cleanup)."""
        if self._synthesis_task:
            self._synthesis_task.cancel()
            try:
                await self._synthesis_task
            except asyncio.CancelledError:
                pass

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
