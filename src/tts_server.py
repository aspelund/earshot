"""
WebSocket server for TTS synthesis using Chatterbox-Turbo.
Receives text, returns WAV audio.
"""
import asyncio
import websockets
import json
from loguru import logger


class TTSServer:
    """
    WebSocket server that receives text and returns synthesized audio.

    Protocol:
    - Client sends: JSON {"type": "synthesize", "text": "...", "request_id": "..."}
    - Client sends: JSON {"type": "cancel", "request_id": "..."} to abort
    - Server sends: Binary WAV audio on success
    - Server sends: JSON {"type": "error", "error": "...", "request_id": "..."} on failure
    """

    def __init__(self, host: str, port: int, tts_handler, auth_token: str = None):
        self.host = host
        self.port = port
        self.tts_handler = tts_handler  # Callable that takes (text, request_id) -> wav_bytes or None
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
                            if not text.strip():
                                await websocket.send(json.dumps({
                                    "type": "error",
                                    "error": "Empty text",
                                    "request_id": request_id
                                }))
                                continue

                            logger.info(f"Synthesizing for {client_addr}: {text[:50]}...")

                            # Mark this request as active
                            self.active_requests[websocket][request_id] = False

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
