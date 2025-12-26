"""
TTS server: Receives text from clients and returns synthesized audio.
Supports multiple backends: Chatterbox-Turbo, Soprano.
"""
import os
import io
import re
import yaml
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
from dotenv import load_dotenv
from loguru import logger

import torch
import soundfile as sf

from .tts_server import TTSServer


# Regex pattern to match emojis and other symbols TTS can't handle
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F000-\U0001F02F"  # mahjong tiles
    "\U0001F0A0-\U0001F0FF"  # playing cards
    "]+",
    flags=re.UNICODE
)


def strip_emojis(text: str) -> str:
    """Remove emojis, markdown formatting, and other symbols that TTS cannot handle."""
    cleaned = EMOJI_PATTERN.sub("", text)
    # Remove markdown formatting characters
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("*", "")
    cleaned = cleaned.replace("_", " ")
    # Clean up any double spaces left behind
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_speakable(text: str) -> bool:
    """Check if text contains enough speakable content."""
    if not text:
        return False
    # Must have at least some alphanumeric characters
    alphanumeric = sum(1 for c in text if c.isalnum())
    return alphanumeric >= 2


def load_cfg():
    path = os.getenv("CONFIG_PATH", "config.tts_server.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_chatterbox_synthesizer(device: str) -> Callable[[str], bytes]:
    """Create a Chatterbox-Turbo synthesizer function."""
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    logger.info(f"Loading Chatterbox-Turbo model on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    logger.info("Chatterbox-Turbo model loaded")

    def synthesize_sync(text: str) -> bytes:
        """Synchronous TTS synthesis using Chatterbox-Turbo."""
        t0 = time.perf_counter()

        with torch.no_grad():
            wav = model.generate(text)

        buffer = io.BytesIO()
        audio_np = wav.squeeze(0).cpu().numpy()
        sf.write(buffer, audio_np, model.sr, format="WAV")
        buffer.seek(0)
        wav_bytes = buffer.read()

        del wav
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        latency = time.perf_counter() - t0
        logger.info(f"[Chatterbox] Synthesized {len(text)} chars in {latency:.2f}s ({len(wav_bytes)} bytes)")

        return wav_bytes

    return synthesize_sync


def create_soprano_synthesizer(device: str) -> Callable[[str], bytes]:
    """Create a Soprano TTS synthesizer function (batch mode)."""
    from soprano import SopranoTTS

    logger.info(f"Loading Soprano TTS model on {device}...")
    model = SopranoTTS(backend="auto", device=device)
    logger.info("Soprano TTS model loaded")

    # Soprano outputs at 32kHz
    sample_rate = 32000

    def synthesize_sync(text: str) -> bytes:
        """Synchronous TTS synthesis using Soprano."""
        t0 = time.perf_counter()

        wav = model.infer(text)

        buffer = io.BytesIO()
        # Soprano returns numpy array directly
        if hasattr(wav, 'cpu'):
            audio_np = wav.squeeze().cpu().numpy()
        else:
            audio_np = wav.squeeze() if hasattr(wav, 'squeeze') else wav
        sf.write(buffer, audio_np, sample_rate, format="WAV")
        buffer.seek(0)
        wav_bytes = buffer.read()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        latency = time.perf_counter() - t0
        logger.info(f"[Soprano] Synthesized {len(text)} chars in {latency:.2f}s ({len(wav_bytes)} bytes)")

        return wav_bytes

    return synthesize_sync


def create_soprano_streaming_synthesizer(device: str):
    """
    Create a Soprano TTS streaming synthesizer.
    Returns (stream_generator_factory, sample_rate).
    """
    import numpy as np
    from soprano import SopranoTTS

    logger.info(f"Loading Soprano TTS model on {device} (streaming mode)...")
    model = SopranoTTS(backend="auto", device=device)
    logger.info("Soprano TTS model loaded (streaming mode)")

    sample_rate = 32000

    def stream_synthesize(text: str):
        """Generator that yields audio chunks as f32 numpy arrays."""
        t0 = time.perf_counter()
        first_chunk = True
        chunk_count = 0

        # Strip emojis before streaming
        clean_text = strip_emojis(text)
        if not is_speakable(clean_text):
            logger.info(f"[Soprano] Skipping non-speakable text: {text!r}")
            return

        for chunk in model.infer_stream(clean_text, chunk_size=1):
            # Convert to f32 numpy array
            if hasattr(chunk, 'cpu'):
                audio_np = chunk.cpu().numpy().astype(np.float32)
            else:
                audio_np = np.asarray(chunk, dtype=np.float32)

            if first_chunk:
                latency = time.perf_counter() - t0
                logger.info(f"[Soprano] First chunk latency: {latency*1000:.1f}ms")
                first_chunk = False

            chunk_count += 1
            yield audio_np

        total_time = time.perf_counter() - t0
        logger.info(f"[Soprano] Streaming complete: {chunk_count} chunks in {total_time:.2f}s")

    return stream_synthesize, sample_rate


def main():
    load_dotenv()
    cfg = load_cfg()

    # Determine device
    device_cfg = cfg.get("tts", {}).get("device", "auto")
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    # Select backend
    backend = cfg.get("tts", {}).get("backend", "chatterbox").lower()
    streaming_enabled = cfg.get("tts", {}).get("streaming", {}).get("enabled", True)

    # Initialize synthesizers based on backend
    stream_handler = None
    stream_sample_rate = 32000

    if backend == "chatterbox":
        synthesize_sync = create_chatterbox_synthesizer(device)
        # Chatterbox doesn't support streaming
    elif backend == "soprano":
        synthesize_sync = create_soprano_synthesizer(device)
        if streaming_enabled:
            stream_handler, stream_sample_rate = create_soprano_streaming_synthesizer(device)
            logger.info("Soprano streaming mode enabled")
    else:
        raise ValueError(f"Unknown TTS backend: {backend}. Supported: chatterbox, soprano")

    # Thread pool for TTS processing (so async handlers don't block)
    executor = ThreadPoolExecutor(max_workers=2)

    async def tts_handler(text: str, request_id: str, should_cancel: Callable[[], bool]) -> Optional[bytes]:
        """
        Synthesize text to audio (batch mode).
        Called from async WebSocket handler - runs TTS in thread pool.
        """
        loop = asyncio.get_event_loop()

        # Strip emojis that TTS cannot handle
        clean_text = strip_emojis(text)
        if not is_speakable(clean_text):
            logger.info(f"Skipping non-speakable text: {text!r} -> {clean_text!r}")
            return None

        try:
            # Run TTS in thread pool to avoid blocking the event loop
            wav_bytes = await loop.run_in_executor(executor, synthesize_sync, clean_text)

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

    server = TTSServer(
        server_host,
        server_port,
        tts_handler,
        auth_token,
        stream_handler=stream_handler,
        stream_sample_rate=stream_sample_rate,
    )

    streaming_status = "streaming" if stream_handler else "batch"
    logger.info(f"Starting TTS server on ws://{server_host}:{server_port} (backend: {backend}, mode: {streaming_status})")
    logger.info("Waiting for clients to connect...")

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
