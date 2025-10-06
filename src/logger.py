"""
Rotating JSONL logger + heartbeat. One JSON per line for downstream agents.
"""
from loguru import logger
import json
import os
import sys
from typing import Dict
from .utils_time import utc_now_iso


class JSONLLogger:
    def __init__(self, path: str, rotate_mb: int, backups: int, heartbeat_s: int):
        self.path = path
        self.heartbeat_s = heartbeat_s

        # Configure loguru for rotating JSONL output
        logger.remove()  # Remove default handler
        logger.add(
            path,
            rotation=f"{rotate_mb} MB",
            retention=backups,
            format="{message}",  # Raw message only (we'll provide JSON)
            enqueue=True,  # Thread-safe
            serialize=False  # We handle JSON ourselves
        )

        # Also log to stderr for development
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | {message}",
            colorize=True
        )

    def append_segment(self, seg: Dict):
        """Log a transcription segment as JSONL."""
        logger.info(json.dumps(seg, ensure_ascii=False))

    def heartbeat(self):
        """Log a heartbeat message."""
        hb = {
            "type": "heartbeat",
            "ts": utc_now_iso()
        }
        logger.info(json.dumps(hb, ensure_ascii=False))
