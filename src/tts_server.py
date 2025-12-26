"""
WebSocket server for TTS synthesis.
Supports both batch (WAV) and streaming (PCM chunks) modes.
"""
import asyncio
import struct
import websockets
import json
from loguru import logger
from typing import Callable, Generator, Optional


class TTSServer:
    """
    WebSocket server that receives text and returns synthesized audio.

    Protocol (batch mode):
    - Client sends: JSON {"type": "synthesize", "text": "...", "request_id": "..."}
    - Client sends: JSON {"type": "cancel", "request_id": "..."} to abort
    - Server sends: Binary WAV audio on success
    - Server sends: JSON {"type": "error", "error": "...", "request_id": "..."} on failure

    Protocol (streaming mode):
    - Client sends: JSON {"type": "synthesize", "text": "...", "request_id": "...", "stream": true}
    - Server sends: JSON {"type": "stream_start", "request_id": "...", "sample_rate": 32000}
    - Server sends: Binary [4-byte u32 chunk_index][f32 PCM samples...] (multiple)
    - Server sends: JSON {"type": "stream_end", "request_id": "...", "total_chunks": N}
    """

    def __init__(
        self,
        host: str,
        port: int,
        tts_handler: Callable,
        auth_token: str = None,
        stream_handler: Optional[Callable[[str], Generator]] = None,
        stream_sample_rate: int = 32000,
    ):
        self.host = host
        self.port = port
        self.tts_handler = tts_handler  # Callable that takes (text, request_id, should_cancel) -> wav_bytes or None
        self.stream_handler = stream_handler  # Generator factory: (text) -> yields f32 numpy arrays
        self.stream_sample_rate = stream_sample_rate
        self.auth_token = auth_token
        self.server = None
        self.active_clients = set()
        # Track active synthesis tasks per client for cancellation
        self.active_requests = {}  # websocket -> {request_id: should_cancel}

    async def handler(self, websocket):
        """Handle incoming WebSocket connection."""
        client_addr = websocket.remote_address
        logger.info(f"TTS client connected: {client_addr}")

        # Optional authentication
        if self.auth_token:
            try:
                auth_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                if auth_msg != self.auth_token:
                    logger.warning(f"Authentication failed for {client_addr}")
                    await websocket.close(1008, "Authentication failed")
                    return
                logger.info(f"Client {client_addr} authenticated")
            except asyncio.TimeoutError:
                logger.warning(f"Authentication timeout for {client_addr}")
                await websocket.close(1008, "Authentication timeout")
                return

        self.active_clients.add(websocket)
        self.active_requests[websocket] = {}

        try:
            async for message in websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        request_id = data.get("request_id", "unknown")

                        if msg_type == "synthesize":
                            text = data.get("text", "")
                            use_streaming = data.get("stream", False) and self.stream_handler is not None

                            if not text.strip():
                                await websocket.send(json.dumps({
                                    "type": "error",
                                    "error": "Empty text",
                                    "request_id": request_id
                                }))
                                continue

                            logger.info(f"Synthesizing for {client_addr}: {text[:50]}... (stream={use_streaming})")

                            # Mark this request as active
                            self.active_requests[websocket][request_id] = False

                            if use_streaming:
                                # Streaming mode
                                await self._handle_streaming_synthesis(websocket, text, request_id)
                            else:
                                # Batch mode
                                # Create cancel check function
                                def should_cancel():
                                    return self.active_requests.get(websocket, {}).get(request_id, True)

                                # Run TTS
                                wav_bytes = await self.tts_handler(text, request_id, should_cancel)

                                # Clean up request tracking
                                self.active_requests[websocket].pop(request_id, None)

                                if wav_bytes is None:
                                    # Cancelled or failed
                                    if should_cancel():
                                        logger.info(f"Request {request_id} was cancelled")
                                        await websocket.send(json.dumps({
                                            "type": "cancelled",
                                            "request_id": request_id
                                        }))
                                    else:
                                        await websocket.send(json.dumps({
                                            "type": "error",
                                            "error": "Synthesis failed",
                                            "request_id": request_id
                                        }))
                                else:
                                    # Send binary audio
                                    await websocket.send(wav_bytes)
                                    logger.info(f"Sent {len(wav_bytes)} bytes audio to {client_addr}")

                        elif msg_type == "cancel":
                            # Mark request for cancellation
                            if request_id in self.active_requests.get(websocket, {}):
                                self.active_requests[websocket][request_id] = True
                                logger.info(f"Marked request {request_id} for cancellation")

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from {client_addr}: {e}")
                else:
                    logger.warning(f"Received non-text message from {client_addr}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"TTS client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"Error handling TTS client {client_addr}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.active_clients.discard(websocket)
            self.active_requests.pop(websocket, None)
            logger.info(f"TTS client removed: {client_addr}, active clients: {len(self.active_clients)}")

    async def _handle_streaming_synthesis(self, websocket, text: str, request_id: str):
        """Handle streaming TTS synthesis."""
        import queue
        import threading

        client_addr = websocket.remote_address

        try:
            # Send stream start
            await websocket.send(json.dumps({
                "type": "stream_start",
                "request_id": request_id,
                "sample_rate": self.stream_sample_rate,
            }))

            # Use a queue to stream chunks from the generator thread to async handler
            chunk_queue = queue.Queue()
            generation_done = threading.Event()
            generation_error = [None]  # Use list to allow modification in thread

            def generate_chunks():
                """Run generator in separate thread, push chunks to queue."""
                try:
                    for chunk in self.stream_handler(text):
                        chunk_queue.put(chunk)
                except Exception as e:
                    generation_error[0] = e
                finally:
                    generation_done.set()

            # Start generation thread
            gen_thread = threading.Thread(target=generate_chunks, daemon=True)
            gen_thread.start()

            chunk_index = 0
            cancelled = False

            # Stream chunks as they arrive
            while True:
                # Check for cancellation
                if self.active_requests.get(websocket, {}).get(request_id, True):
                    cancelled = True
                    break

                # Try to get a chunk (non-blocking)
                try:
                    audio_chunk = chunk_queue.get(timeout=0.01)

                    # Build binary message: [4-byte index][f32 samples]
                    header = struct.pack('<I', chunk_index)
                    payload = audio_chunk.tobytes()
                    await websocket.send(header + payload)
                    chunk_index += 1

                except queue.Empty:
                    # No chunk available, check if generation is done
                    if generation_done.is_set():
                        # Drain any remaining chunks
                        while not chunk_queue.empty():
                            audio_chunk = chunk_queue.get_nowait()
                            header = struct.pack('<I', chunk_index)
                            payload = audio_chunk.tobytes()
                            await websocket.send(header + payload)
                            chunk_index += 1
                        break
                    # Otherwise keep waiting
                    await asyncio.sleep(0.001)

            # Check for generation error
            if generation_error[0] is not None:
                raise generation_error[0]

            # Clean up request tracking
            self.active_requests[websocket].pop(request_id, None)

            if cancelled:
                logger.info(f"Stream {request_id} was cancelled after {chunk_index} chunks")
                await websocket.send(json.dumps({
                    "type": "cancelled",
                    "request_id": request_id
                }))
            else:
                # Send stream end
                await websocket.send(json.dumps({
                    "type": "stream_end",
                    "request_id": request_id,
                    "total_chunks": chunk_index,
                }))
                logger.info(f"Streamed {chunk_index} chunks to {client_addr}")

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.active_requests[websocket].pop(request_id, None)
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e),
                "request_id": request_id
            }))

    async def start_server(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
            max_size=10*1024*1024  # 10MB max message size
        )
        logger.info(f"TTS server listening on ws://{self.host}:{self.port}")
        await asyncio.Future()  # Run forever

    def run(self):
        """Run the WebSocket server (blocking)."""
        asyncio.run(self.start_server())
