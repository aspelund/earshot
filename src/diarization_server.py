"""
Combined HTTP + WebSocket server using aiohttp.
- POST /transcribe - File upload for diarized transcription
- GET  /ws/stream  - Real-time WebSocket streaming (existing protocol)
- GET  /health     - Health check
"""
import os
import io
import json
import struct
import time
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
import numpy as np
from loguru import logger


class DiarizationServer:
    """
    Server that provides both HTTP file upload for batch diarized transcription
    and WebSocket streaming for real-time transcription.
    """

    def __init__(self, host: str, port: int, cfg: dict, auth_token: str = None):
        self.host = host
        self.port = port
        self.cfg = cfg
        self.auth_token = auth_token
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Lazy-loaded models
        self._whisper_stt = None
        self._parakeet_stt = None
        self._diarizer = None

        # WebSocket state
        self.active_ws_clients = set()

    def _get_stt(self, model: str):
        """Get or lazily load the requested STT model."""
        if model == "parakeet":
            if self._parakeet_stt is None:
                from .stt_parakeet import ParakeetSTT
                logger.info("Loading Parakeet STT model...")
                self._parakeet_stt = ParakeetSTT(self.cfg["stt"])
                logger.info("Parakeet STT model loaded")
            return self._parakeet_stt
        else:  # whisper
            if self._whisper_stt is None:
                from .stt_whisper import FastSTT
                logger.info("Loading Whisper STT model...")
                self._whisper_stt = FastSTT(self.cfg["stt"])
                logger.info("Whisper STT model loaded")
            return self._whisper_stt

    def _get_diarizer(self):
        """Get or lazily load the diarizer."""
        if self._diarizer is None:
            from .diarization import SpeakerDiarizer
            logger.info("Loading speaker diarization model...")
            self._diarizer = SpeakerDiarizer(self.cfg.get("diarization", {}))
            logger.info("Speaker diarization model loaded")
        return self._diarizer

    def _load_audio_as_pcm16(self, audio_bytes: bytes) -> tuple:
        """
        Load audio from bytes and convert to 16kHz mono PCM16.
        Returns: (pcm16_bytes, duration_s)
        """
        import soundfile as sf

        # Try soundfile first (handles wav, flac, ogg)
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes))

            # Stereo to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample to 16kHz if needed
            if sr != 16000:
                import scipy.signal
                num_samples = int(len(audio) * 16000 / sr)
                audio = scipy.signal.resample(audio, num_samples)

            # Convert to PCM16
            pcm16 = (audio * 32767).astype(np.int16)
            return pcm16.tobytes(), len(pcm16) / 16000

        except Exception as e:
            logger.debug(f"soundfile failed, trying pydub: {e}")

        # Fall back to pydub for mp3, m4a, etc.
        try:
            from pydub import AudioSegment
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            segment = segment.set_frame_rate(16000).set_channels(1)
            pcm16 = np.array(segment.get_array_of_samples(), dtype=np.int16)
            return pcm16.tobytes(), len(pcm16) / 16000
        except Exception as e:
            raise ValueError(f"Could not decode audio: {e}")

    def _run_asr(self, pcm16_bytes: bytes, model: str, language: str) -> dict:
        """Run ASR transcription (called in thread pool)."""
        stt = self._get_stt(model)
        result = stt.transcribe(
            pcm16_bytes,
            language=language,
            beam_size=self.cfg["stt"].get("beam_size", 1),
            word_timestamps=True
        )
        return result

    def _run_diarization(
        self,
        audio_path: str,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> list:
        """Run speaker diarization (called in thread pool)."""
        diarizer = self._get_diarizer()
        return diarizer.diarize(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

    async def handle_transcribe(self, request: web.Request) -> web.Response:
        """Handle POST /transcribe for file upload with diarization."""
        from .transcript_merger import merge_asr_diarization, format_transcript

        t0 = time.perf_counter()

        # Parse multipart form
        try:
            reader = await request.multipart()
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": f"Invalid request: {e}"},
                status=400
            )

        audio_data = None
        model = self.cfg["stt"].get("backend", "whisper")
        language = self.cfg["stt"].get("language")
        num_speakers = None
        min_speakers = None
        max_speakers = None

        async for field in reader:
            if field.name == "audio":
                audio_data = await field.read()
            elif field.name == "model":
                model = (await field.read()).decode().strip()
            elif field.name == "language":
                language = (await field.read()).decode().strip() or None
            elif field.name == "num_speakers":
                try:
                    num_speakers = int((await field.read()).decode())
                except ValueError:
                    pass
            elif field.name == "min_speakers":
                try:
                    min_speakers = int((await field.read()).decode())
                except ValueError:
                    pass
            elif field.name == "max_speakers":
                try:
                    max_speakers = int((await field.read()).decode())
                except ValueError:
                    pass

        if not audio_data:
            return web.json_response(
                {"status": "error", "error": "No audio file provided"},
                status=400
            )

        # Validate model
        if model not in ("whisper", "parakeet"):
            return web.json_response(
                {"status": "error", "error": f"Invalid model: {model}. Use 'whisper' or 'parakeet'"},
                status=400
            )

        # Save to temp file (Pyannote needs a file path)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            # Convert to PCM16 and save as WAV for diarization
            try:
                pcm16_bytes, duration_s = self._load_audio_as_pcm16(audio_data)
            except ValueError as e:
                return web.json_response(
                    {"status": "error", "error": str(e)},
                    status=400
                )

            # Write WAV file for Pyannote
            import soundfile as sf
            pcm16_array = np.frombuffer(pcm16_bytes, dtype=np.int16)
            audio_float = pcm16_array.astype(np.float32) / 32768.0
            sf.write(tmp_path, audio_float, 16000)

            logger.info(f"Processing {duration_s:.1f}s audio with model={model}, language={language}")

            # Run ASR and diarization in parallel
            loop = asyncio.get_event_loop()

            asr_task = loop.run_in_executor(
                self.executor,
                self._run_asr,
                pcm16_bytes, model, language
            )
            diar_task = loop.run_in_executor(
                self.executor,
                self._run_diarization,
                tmp_path, num_speakers, min_speakers, max_speakers
            )

            try:
                asr_result, diar_result = await asyncio.gather(asr_task, diar_task)
            except Exception as e:
                logger.error(f"Processing failed: {e}")
                return web.json_response(
                    {"status": "error", "error": f"Processing failed: {e}"},
                    status=500
                )

            # Merge results
            utterances = merge_asr_diarization(asr_result.get("words", []), diar_result)
            formatted = format_transcript(utterances)

            processing_time = time.perf_counter() - t0
            num_speakers_detected = len(set(u["speaker"] for u in utterances)) if utterances else 0

            logger.info(f"Transcription complete: {num_speakers_detected} speakers, {len(utterances)} utterances, {processing_time:.2f}s")

            return web.json_response({
                "status": "success",
                "duration_s": round(duration_s, 2),
                "processing_time_s": round(processing_time, 2),
                "num_speakers": num_speakers_detected,
                "transcript": utterances,
                "formatted": formatted
            })

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def handle_ws_stream(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time streaming."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_addr = request.remote
        logger.info(f"WebSocket client connected: {client_addr}")

        # Optional authentication
        if self.auth_token:
            try:
                auth_msg = await asyncio.wait_for(ws.receive_str(), timeout=5.0)
                if auth_msg != self.auth_token:
                    logger.warning(f"Authentication failed for {client_addr}")
                    await ws.close(code=1008, message=b"Authentication failed")
                    return ws
                logger.info(f"Client {client_addr} authenticated")
            except asyncio.TimeoutError:
                logger.warning(f"Authentication timeout for {client_addr}")
                await ws.close(code=1008, message=b"Authentication timeout")
                return ws

        self.active_ws_clients.add(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    message = msg.data

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
                        start_iso = header.get("start_utc", "")
                        end_iso = header.get("end_utc", "")
                        language = header.get("language")
                        model = header.get("model", self.cfg["stt"].get("backend", "whisper"))

                        duration_s = len(pcm_bytes) / 2 / 16000
                        logger.info(f"Received segment: {duration_s:.2f}s, model={model}")

                        # Run transcription in thread pool
                        loop = asyncio.get_event_loop()
                        t0 = time.perf_counter()

                        result = await loop.run_in_executor(
                            self.executor,
                            self._run_asr,
                            pcm_bytes, model, language
                        )

                        latency = round(time.perf_counter() - t0, 3)

                        # Build response
                        from datetime import datetime
                        if start_iso:
                            start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                            start_ms = int(start_dt.timestamp() * 1000)
                        else:
                            start_ms = 0

                        tokens = []
                        for w in result.get("words", []):
                            tokens.append({
                                "w": w["w"],
                                "start_ms": start_ms + int(w["start_s"] * 1000),
                                "end_ms": start_ms + int(w["end_s"] * 1000)
                            })

                        response = {
                            "type": "asr_segment",
                            "start_utc": start_iso,
                            "end_utc": end_iso,
                            "latency_s": latency,
                            "text": result.get("text", ""),
                            "tokens": tokens
                        }

                        await ws.send_json(response)
                        logger.info(f"Transcribed: {result.get('text', '')[:50]}... ({latency:.2f}s)")

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON header: {e}")
                    except Exception as e:
                        logger.error(f"Error processing segment: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")

        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.active_ws_clients.discard(ws)
            logger.info(f"WebSocket client disconnected: {client_addr}")

        return ws

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "ok",
            "active_ws_clients": len(self.active_ws_clients)
        })

    def create_app(self) -> web.Application:
        """Create the aiohttp application."""
        max_size = self.cfg.get("http", {}).get("max_file_size_mb", 100) * 1024 * 1024
        app = web.Application(client_max_size=max_size)

        # Routes
        app.router.add_post("/transcribe", self.handle_transcribe)
        app.router.add_get("/ws/stream", self.handle_ws_stream)
        app.router.add_get("/health", self.handle_health)

        return app

    def run(self):
        """Run the server (blocking)."""
        app = self.create_app()
        web.run_app(app, host=self.host, port=self.port)
