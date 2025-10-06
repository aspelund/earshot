"""
WebSocket server for receiving audio streams from remote clients.
Integrates with the VAD->STT pipeline by providing a NetworkStream.
Sends transcription results back to clients.
"""
import asyncio
import websockets
import threading
import json
from loguru import logger
from .audio_network import NetworkStream


class AudioStreamServer:
    """
    WebSocket server that receives PCM16 audio and feeds it to a NetworkStream.
    Sends transcription results back to connected clients.
    """

    def __init__(self, host: str, port: int, network_stream: NetworkStream, auth_token: str = None):
        self.host = host
        self.port = port
        self.network_stream = network_stream
        self.auth_token = auth_token
        self.server = None
        self.active_clients = set()
        self.event_loop = None

    async def handler(self, websocket):
        """Handle incoming WebSocket connection."""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

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
            # Receive audio chunks and push to NetworkStream
            async for message in websocket:
                if isinstance(message, bytes):
                    # Push PCM16 audio data to the stream
                    self.network_stream.push_audio(message)
                else:
                    logger.warning(f"Received non-binary message from {client_addr}: {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}")
        finally:
            self.active_clients.discard(websocket)
            logger.info(f"Client removed: {client_addr}, active clients: {len(self.active_clients)}")

    async def start_server(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        logger.info(f"Audio stream server listening on ws://{self.host}:{self.port}")
        await asyncio.Future()  # Run forever

    def run_in_thread(self):
        """Run the WebSocket server in a background thread."""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.event_loop = loop
            loop.run_until_complete(self.start_server())

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        logger.info("WebSocket server thread started")
        # Give the loop time to start
        import time
        time.sleep(0.1)
        return thread

    def broadcast_transcription(self, transcription_data: dict):
        """
        Send transcription result to all connected clients.
        Called from the STT worker thread.
        """
        if not self.active_clients or not self.event_loop:
            return

        # Create JSON message
        message = json.dumps(transcription_data)

        # Schedule the broadcast in the event loop
        asyncio.run_coroutine_threadsafe(
            self._broadcast_async(message),
            self.event_loop
        )

    async def _broadcast_async(self, message: str):
        """Actually send the message to all clients (runs in event loop)."""
        if not self.active_clients:
            return

        # Send to all connected clients
        disconnected = set()
        for client in self.active_clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.error(f"Error sending to client {client.remote_address}: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.active_clients -= disconnected
