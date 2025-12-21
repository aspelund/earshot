"""
src2 - Improved conversational client components with robust cancellation.

Key improvements over src/:
- asyncio.Event instead of boolean flags for thread-safe cancellation
- Timeout-based network calls to break blocking operations
- Generation counters in all components for stale data filtering
"""

from .audio import AudioPlayer
from .services import LLMClient, TTSClient

__all__ = ["AudioPlayer", "LLMClient", "TTSClient"]
