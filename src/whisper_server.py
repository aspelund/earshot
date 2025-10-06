"""
WebSocket server for receiving pre-segmented audio from VAD clients.
Runs STT only (no VAD) and sends transcriptions back.
"""
import asyncio
import websockets
import json
import struct
from loguru import logger


class WhisperServer:
    """
    WebSocket server that receives pre-segmented PCM16 audio from VAD clients,
    transcribes it with Whisper, and sends results back.

    Protocol:
    - Client sends: JSON header (length as 4-byte uint32) + PCM16 audio data
    - Server sends: JSON transcription result
    """

    def __init__(self, host: str, port: int, stt_handler, auth_token: str = None):
        self.host = host
        self.port = port
        self.stt_handler = stt_handler  # Callable that takes (pcm_bytes, start_iso, end_iso) -> transcription_dict
        self.auth_token = auth_token
        self.server = None
        self.active_clients = set()

    async def handler(self, websocket):
        """Handle incoming WebSocket connection."""
        client_addr = websocket.remote_address
        logger.info(f"VAD client connected: {client_addr}")

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

        try:
            # Receive segments and transcribe
            async for message in websocket:
                if isinstance(message, bytes):
                    # Parse protocol: 4-byte header length + JSON header + PCM16 data
                    if len(message) < 4:
                        logger.warning(f"Message too short from {client_addr}")
                        continue

                    # Read header length (uint32 big-endian)
                    header_len = struct.unpack('>I', message[:4])[0]

                    if len(message) < 4 + header_len:
                        logger.warning(f"Invalid message format from {client_addr}")
                        continue

                    # Parse JSON header
                    header_bytes = message[4:4+header_len]
                    pcm_bytes = message[4+header_len:]

                    try:
                        header = json.loads(header_bytes.decode('utf-8'))
                        start_iso = header.get("start_utc")
                        end_iso = header.get("end_utc")

                        duration_s = len(pcm_bytes) / 2 / 16000
                        logger.info(f"Received segment from {client_addr}: {duration_s:.2f}s, transcribing...")

                        # Transcribe the segment
                        result = await self.stt_handler(pcm_bytes, start_iso, end_iso)

                        # Send result back to client
                        await websocket.send(json.dumps(result))
                        logger.info(f"Sent transcription back to {client_addr}")

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON header from {client_addr}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing segment from {client_addr}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

                else:
                    logger.warning(f"Received non-binary message from {client_addr}: {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"VAD client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"Error handling VAD client {client_addr}: {e}")
        finally:
            self.active_clients.discard(websocket)
            logger.info(f"VAD client removed: {client_addr}, active clients: {len(self.active_clients)}")

    async def start_server(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
            max_size=10*1024*1024  # 10MB max message size (for long audio segments)
        )
        logger.info(f"Whisper server listening on ws://{self.host}:{self.port}")
        await asyncio.Future()  # Run forever

    def run(self):
        """Run the WebSocket server (blocking)."""
        asyncio.run(self.start_server())
