"""
TTS server: Receives text from clients and returns synthesized audio.
Uses Chatterbox-Turbo for synthesis.
"""
import os
import io
import yaml
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
from dotenv import load_dotenv
from loguru import logger

import torch
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS

from .tts_server import TTSServer


def load_cfg():
    path = os.getenv("CONFIG_PATH", "config.tts_server.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    load_dotenv()
    cfg = load_cfg()

    # Determine device
    device_cfg = cfg.get("tts", {}).get("device", "auto")
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    # Initialize Chatterbox-Turbo
    logger.info(f"Loading Chatterbox-Turbo model on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    logger.info("Chatterbox-Turbo model loaded")

    # Thread pool for TTS processing (so async handlers don't block)
    executor = ThreadPoolExecutor(max_workers=2)

    def synthesize_sync(text: str) -> bytes:
        """Synchronous TTS synthesis - runs in thread pool."""
        t0 = time.perf_counter()

        # Generate audio
        with torch.no_grad():
            wav = model.generate(text)

        # Convert tensor to WAV bytes
        buffer = io.BytesIO()
        audio_np = wav.squeeze(0).cpu().numpy()
        sf.write(buffer, audio_np, model.sr, format="WAV")
        buffer.seek(0)
        wav_bytes = buffer.read()

        # Clear GPU cache to prevent memory fragmentation
        del wav
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        latency = time.perf_counter() - t0
        logger.info(f"Synthesized {len(text)} chars in {latency:.2f}s ({len(wav_bytes)} bytes)")

        return wav_bytes

    async def tts_handler(text: str, request_id: str, should_cancel: Callable[[], bool]) -> Optional[bytes]:
        """
        Synthesize text to audio.
        Called from async WebSocket handler - runs TTS in thread pool.
        """
        loop = asyncio.get_event_loop()

        try:
            # Run TTS in thread pool to avoid blocking the event loop
            wav_bytes = await loop.run_in_executor(executor, synthesize_sync, text)

            # Check if cancelled after synthesis
            if should_cancel():
                return None

            return wav_bytes

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    # Start TTS server
    server_host = cfg.get("server", {}).get("host", "0.0.0.0")
    server_port = cfg.get("server", {}).get("port", 8766)
    auth_token = cfg.get("server", {}).get("auth_token")

    server = TTSServer(server_host, server_port, tts_handler, auth_token)

    logger.info(f"Starting TTS server on ws://{server_host}:{server_port}")
    logger.info("Waiting for clients to connect...")

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
