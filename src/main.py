"""
Wire: Audio Source -> TEN VAD -> Segmentor -> STT -> JSONL logger (+ heartbeat).
All real-time work must not block the audio callback.
STT runs in a separate worker thread with a queue.

Audio sources:
- MicStream: Local microphone capture
- NetworkStream: Receive audio from remote client via WebSocket
"""
import os
import yaml
import time
import threading
from queue import Queue
from dotenv import load_dotenv
from .audio_network import NetworkStream
from .stream_server import AudioStreamServer
from .vad_ten import TenVAD
from .segmentor import Segmentor
from .stt_whisper import FastSTT
from .logger import JSONLLogger
from .utils_time import utc_now_iso


def load_cfg():
    path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    load_dotenv()
    cfg = load_cfg()

    # Ensure log dir exists
    log_dir = os.path.expanduser(cfg["logging"]["dir"])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, cfg["logging"]["file"])

    # Init components
    print("Initializing components...")

    # Select audio source based on config
    audio_source = cfg["audio"].get("source", "mic")

    if audio_source == "network":
        # Create network stream and start WebSocket server
        print("Audio source: Network stream")
        audio_stream = NetworkStream(
            cfg["audio"]["sample_rate"],
            cfg["audio"]["channels"],
            cfg["audio"]["frame_ms"]
        )

        # Start WebSocket server to receive audio
        server_host = cfg["network"].get("host", "0.0.0.0")
        server_port = cfg["network"].get("port", 8765)
        auth_token = cfg["network"].get("auth_token")

        stream_server = AudioStreamServer(server_host, server_port, audio_stream, auth_token)
        stream_server.run_in_thread()
        print(f"WebSocket server listening on ws://{server_host}:{server_port}")

    else:
        # Use local microphone (lazy import to avoid PortAudio dependency when not needed)
        print("Audio source: Local microphone")
        from .audio import MicStream
        audio_stream = MicStream(
            cfg["audio"]["sample_rate"],
            cfg["audio"]["channels"],
            cfg["audio"]["frame_ms"]
        )

    vad = TenVAD(cfg["vad"]["model_path"])
    seg = Segmentor(cfg)
    stt = FastSTT(cfg["stt"])
    jl = JSONLLogger(log_path, cfg["logging"]["rotate_mb"], cfg["logging"]["backups"], cfg["logging"]["heartbeat_s"])

    # Queue for segments to transcribe
    segment_queue = Queue(maxsize=10)  # Max 10 pending segments

    print(f"Listening... (logs: {log_path})")

    # STT worker thread
    def stt_worker():
        from datetime import datetime, timezone
        while True:
            # Block until segment available
            item = segment_queue.get()
            if item is None:  # Poison pill to stop
                break

            pcm_bytes, start_iso, end_iso = item

            # Transcribe
            t0 = time.perf_counter()
            out = stt.transcribe(
                pcm_bytes,
                language=cfg["stt"]["language"],
                beam_size=cfg["stt"]["beam_size"],
                word_timestamps=cfg["stt"]["word_timestamps"]
            )
            latency = round(time.perf_counter() - t0, 3)

            # Convert word timestamps to absolute UTC milliseconds
            start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            start_ms = int(start_dt.timestamp() * 1000)

            tokens = []
            for w in out["words"]:
                tokens.append({
                    "w": w["w"],
                    "start_ms": start_ms + int(w["start_s"] * 1000),
                    "end_ms": start_ms + int(w["end_s"] * 1000)
                })

            seg_json = {
                "type": "asr_segment",
                "start_utc": start_iso,
                "end_utc": end_iso,
                "latency_s": latency,
                "text": out["text"],
                "tokens": tokens
            }
            jl.append_segment(seg_json)

            # Print transcription to terminal
            print(out["text"])

            segment_queue.task_done()

    # Heartbeat thread
    def heart():
        while True:
            time.sleep(cfg["logging"]["heartbeat_s"])
            jl.heartbeat()

    threading.Thread(target=heart, daemon=True).start()
    threading.Thread(target=stt_worker, daemon=True).start()

    # Main loop (never blocks on STT)
    try:
        for frame in audio_stream.frames():
            p = vad.prob_speech(frame)  # float 0..1
            flush = seg.update(frame, p)

            if flush:
                # Queue segment for async transcription
                try:
                    segment_queue.put_nowait(flush)
                except:
                    # Queue full - drop segment (shouldn't happen with base.en)
                    print("⚠ STT queue full, dropping segment")

    except KeyboardInterrupt:
        print("\nShutting down...")
        audio_stream.close()
        segment_queue.put(None)  # Stop worker thread


if __name__ == "__main__":
    main()
