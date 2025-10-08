"""Client for LLM (Large Language Model) service"""
import sys
import asyncio
import aiohttp
from typing import Optional, List, Dict


class LLMClient:
    """Single-request LLM client with streaming sentence support"""

    def __init__(self, host: str, port: int, model: str, temperature: float, max_tokens: int):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session: Optional[aiohttp.ClientSession] = None

        # Current request state
        self.current_request: Optional[List[Dict[str, str]]] = None
        self.pending_sentences: List[str] = []

        # Control flags
        self.is_generating = False
        self.should_abort = False
        self._processing_task: Optional[asyncio.Task] = None
        self._started = False

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def start(self) -> None:
        """Start the processing loop (call once inside async context)"""
        if not self._started:
            self._processing_task = asyncio.create_task(self._processing_loop())
            self._started = True

    def enqueue(self, messages: List[Dict[str, str]]) -> None:
        """
        Set new LLM request (replaces any in-progress).
        Non-blocking.
        """
        if not self._started:
            raise RuntimeError("LLMClient not started. Call start() first in async context.")

        # Abort any current request
        if self.current_request is not None:
            self.should_abort = True

        # Set new request
        self.current_request = messages
        self.should_abort = False

    def get_ready_sentences(self) -> List[str]:
        """
        Get all completed sentences and clear the list (non-blocking).
        Returns list of sentence strings.
        """
        if not self.pending_sentences:
            return []

        result = self.pending_sentences.copy()
        self.pending_sentences.clear()
        return result

    def is_processing(self) -> bool:
        """Returns True if currently generating or has pending sentences"""
        return self.is_generating or len(self.pending_sentences) > 0 or self.current_request is not None

    def abort(self) -> None:
        """Cancel current request and clear pending sentences (non-blocking)"""
        self.current_request = None
        self.pending_sentences.clear()
        self.should_abort = True

    async def _processing_loop(self):
        """Background task that processes LLM requests"""
        while True:
            try:
                # Wait for a request
                if self.current_request is None:
                    await asyncio.sleep(0.01)
                    continue

                # Process the request
                messages = self.current_request
                self.current_request = None  # Clear immediately so new requests can come in

                await self._generate(messages)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"LLM processing loop error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    async def _generate(self, messages: List[Dict[str, str]]):
        """Generate LLM response with streaming"""
        if not self.session:
            raise RuntimeError("LLMClient not initialized. Use 'async with' context manager.")

        self.is_generating = True
        self.should_abort = False

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True  # Enable streaming
        }

        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response.raise_for_status()

                # Stream the response
                accumulated_text = ""
                async for line in response.content:
                    if self.should_abort:
                        break

                    # Parse SSE format
                    line = line.decode('utf-8').strip()
                    if not line or not line.startswith('data: '):
                        continue

                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break

                    try:
                        import json
                        data = json.loads(data_str)

                        # Extract content delta
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')

                            if content:
                                accumulated_text += content

                                # Check for sentence boundaries
                                sentences = self._extract_sentences(accumulated_text)
                                if sentences:
                                    for sentence in sentences[:-1]:  # All but last (might be incomplete)
                                        if sentence.strip():
                                            self.pending_sentences.append(sentence.strip())
                                    accumulated_text = sentences[-1]  # Keep incomplete part

                    except json.JSONDecodeError:
                        continue

                # Add any remaining text as final sentence
                if accumulated_text.strip() and not self.should_abort:
                    self.pending_sentences.append(accumulated_text.strip())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"LLM API error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            self.is_generating = False

    def _extract_sentences(self, text: str) -> List[str]:
        """
        Split text on sentence boundaries.
        Returns list where last item might be incomplete.
        """
        # Simple sentence splitting on .!?
        import re
        # Split but keep delimiters
        parts = re.split(r'([.!?]+)', text)

        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            sentences.append(sentence + punct)

        # Add remaining text (incomplete sentence)
        if len(parts) % 2 == 1:
            sentences.append(parts[-1])
        elif not sentences:
            sentences.append(text)

        return sentences

    async def stop(self):
        """Stop the processing loop (cleanup)"""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
