"""
ASR server: Receives pre-segmented audio from VAD clients and transcribes.
Supports multiple backends: Whisper (default) or NVIDIA Parakeet.
No VAD or segmentation - just STT on incoming segments.
"""
import os

# Disable CUDA graphs for NeMo (must be set before importing NeMo)
os.environ["NEMO_DISABLE_CUDA_GRAPH_DECODER"] = "1"

import yaml
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from .whisper_server import WhisperServer
from loguru import logger


def load_cfg():
    path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    load_dotenv()
    cfg = load_cfg()

    # Select STT backend
    backend = cfg["stt"].get("backend", "whisper")

    if backend == "parakeet":
        from .stt_parakeet import ParakeetSTT
        logger.info("Initializing NVIDIA Parakeet STT...")
        stt = ParakeetSTT(cfg["stt"])
        logger.info(f"Parakeet model loaded: {cfg['stt'].get('model_name', 'nvidia/parakeet-tdt-0.6b-v2')}")
    else:
        from .stt_whisper import FastSTT
        logger.info("Initializing Whisper STT...")
        stt = FastSTT(cfg["stt"])
        logger.info(f"Whisper model loaded: {cfg['stt'].get('model_size', 'medium')}")

    # Thread pool for STT processing (so async handlers don't block)
    executor = ThreadPoolExecutor(max_workers=4)

    async def stt_handler(pcm_bytes: bytes, start_iso: str, end_iso: str, language: str = None) -> dict:
        """
        Transcribe a pre-segmented audio chunk.
        Called from async WebSocket handler - runs STT in thread pool.

        Args:
            language: Optional language code from client (e.g., "en", "sv").
                      Falls back to config if None.
        """
        loop = asyncio.get_event_loop()

        # Use client language hint if provided, otherwise fall back to config
        target_language = language if language else cfg["stt"]["language"]

        # Run STT in thread pool to avoid blocking the event loop
        t0 = time.perf_counter()
        result = await loop.run_in_executor(
            executor,
            stt.transcribe,
            pcm_bytes,
            target_language,
            cfg["stt"]["beam_size"],
            cfg["stt"]["word_timestamps"]
        )
        latency = round(time.perf_counter() - t0, 3)

        # Convert word timestamps to absolute UTC milliseconds
        from datetime import datetime, timezone
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        start_ms = int(start_dt.timestamp() * 1000)

        tokens = []
        for w in result["words"]:
            tokens.append({
                "w": w["w"],
                "start_ms": start_ms + int(w["start_s"] * 1000),
                "end_ms": start_ms + int(w["end_s"] * 1000)
            })

        transcription = {
            "type": "asr_segment",
            "start_utc": start_iso,
            "end_utc": end_iso,
            "latency_s": latency,
            "text": result["text"],
            "tokens": tokens
        }

        # Log to console
        logger.info(f"Transcribed: {result['text']} (latency: {latency:.2f}s)")

        return transcription

    # Start Whisper server
    server_host = cfg["network"].get("host", "0.0.0.0")
    server_port = cfg["network"].get("port", 8765)
    auth_token = cfg["network"].get("auth_token")

    server = WhisperServer(server_host, server_port, stt_handler, auth_token)

    logger.info(f"Starting Whisper server on ws://{server_host}:{server_port}")
    logger.info("Waiting for VAD clients to connect...")

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
