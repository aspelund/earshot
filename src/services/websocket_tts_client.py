"""Client for WebSocket TTS server"""
import sys
import asyncio
import json
import uuid
from typing import Optional, List

import websockets


class WebSocketTTSClient:
    """Queue-based TTS client that connects to TTS WebSocket server"""

    def __init__(self, url: str, auth_token: str = None):
        self.url = url
        self.auth_token = auth_token
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

        # Queue for text to synthesize
        self.text_queue: Optional[asyncio.Queue] = None

        # List of completed audio ready for pickup
        self.ready_audio: List[bytes] = []

        # Track pending requests for cancellation
        self.pending_requests: dict = {}  # request_id -> text

        # Control flags
        self.should_stop = False
        self._synthesis_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._started = False
        self._connected = False

        # Generation counter for abort handling
        # Incremented on abort to invalidate all in-flight responses
        self.generation = 0
        self.active_generation = 0

    async def __aenter__(self):
        # Connect to server
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
        if self.websocket:
            await self.websocket.close()
            self._connected = False

    def start(self) -> None:
        """Start the synthesis loop (call once inside async context)"""
        if not self._started:
            self.text_queue = asyncio.Queue()
            self._synthesis_task = asyncio.create_task(self._synthesis_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._started = True

    def enqueue(self, text: str) -> None:
        """Add text to synthesis queue (non-blocking)"""
        if not self._started:
            raise RuntimeError("WebSocketTTSClient not started. Call start() first in async context.")
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
        """Returns True if has queued text, pending requests, or ready audio"""
        has_queued_text = self.text_queue and not self.text_queue.empty()
        has_pending = len(self.pending_requests) > 0
        has_ready_audio = len(self.ready_audio) > 0
        return has_queued_text or has_pending or has_ready_audio

    def abort(self) -> None:
        """Clear text queue, cancel pending requests, and clear ready audio (non-blocking)"""
        # Clear text queue
        if self.text_queue:
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Send cancel for all pending requests
        if self._connected and self.websocket:
            for request_id in list(self.pending_requests.keys()):
                try:
                    # Schedule cancel message (non-blocking)
                    asyncio.create_task(self._send_cancel(request_id))
                except Exception:
                    pass

        # Clear pending requests
        self.pending_requests.clear()

        # Clear ready audio
        self.ready_audio.clear()

        # Set stop flag
        self.should_stop = True

        # Increment generation to invalidate any in-flight responses
        self.generation += 1

    async def _send_cancel(self, request_id: str):
        """Send cancel message to server"""
        try:
            await self.websocket.send(json.dumps({
                "type": "cancel",
                "request_id": request_id
            }))
        except Exception:
            pass

    async def _synthesis_loop(self):
        """Background task that sends text to server"""
        while True:
            try:
                # Wait for next text item
                text = await self.text_queue.get()

                # Capture current generation for this synthesis batch
                # Late responses from previous generations will be discarded
                self.active_generation = self.generation

                # Generate request ID
                request_id = str(uuid.uuid4())

                # Track pending request
                self.pending_requests[request_id] = text

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
        """Background task that receives audio from server"""
        while True:
            try:
                if not self._connected or not self.websocket:
                    await asyncio.sleep(0.1)
                    continue

                message = await self.websocket.recv()

                if isinstance(message, bytes):
                    # Binary audio data - only accept if from current generation
                    # This filters out stale audio from before an abort
                    if self.active_generation == self.generation:
                        self.ready_audio.append(message)
                else:
                    # JSON response (error or cancelled)
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
                break
            except Exception as e:
                print(f"Receive loop error: {e}", file=sys.stderr)

    async def stop(self):
        """Stop the synthesis loop (cleanup)"""
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
