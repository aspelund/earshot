"""
Diarized transcription server entry point.
Provides both HTTP file upload and WebSocket streaming endpoints.

Endpoints:
  POST /transcribe  - Upload audio file, get diarized transcript
  GET  /ws/stream   - Real-time WebSocket streaming
  GET  /health      - Health check
"""
import os

# Disable CUDA graphs for NeMo (must be set before importing NeMo)
os.environ["NEMO_DISABLE_CUDA_GRAPH_DECODER"] = "1"

import yaml
from dotenv import load_dotenv
from loguru import logger
from .diarization_server import DiarizationServer


def load_cfg():
    path = os.getenv("CONFIG_PATH", "config.yaml")
    path = os.path.abspath(path)
    logger.info(f"Loading config from: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    load_dotenv()
    cfg = load_cfg()

    host = cfg.get("network", {}).get("host", "0.0.0.0")
    port = cfg.get("network", {}).get("port", 8765)
    auth_token = cfg.get("network", {}).get("auth_token")

    logger.info("=" * 60)
    logger.info("Diarization Server")
    logger.info("=" * 60)
    logger.info(f"Listening on http://{host}:{port}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  POST /transcribe  - Upload audio for diarized transcription")
    logger.info(f"  GET  /ws/stream   - Real-time WebSocket streaming")
    logger.info(f"  GET  /health      - Health check")
    logger.info("")
    logger.info(f"Default STT backend: {cfg.get('stt', {}).get('backend', 'whisper')}")
    logger.info(f"Diarization model: {cfg.get('diarization', {}).get('model', 'pyannote/speaker-diarization-3.1')}")
    logger.info("=" * 60)

    server = DiarizationServer(host, port, cfg, auth_token)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
