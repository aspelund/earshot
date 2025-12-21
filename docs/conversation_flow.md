# Conversational Client Architecture

The conversational client (`scripts/conversational_client.py`) provides robust voice conversation with:
- **asyncio.Event** instead of boolean flags for thread-safe signaling
- **Timeout-based recv()** to break blocking network calls every 100ms
- **Generation counters** in all components to filter stale data
- **Atomic interrupt handling** with queue purging

```bash
# To run:
bash scripts/run_whisper_server.sh   # Terminal 1
bash scripts/run_tts_server.sh       # Terminal 2
bash scripts/start_conversation.sh   # Terminal 3
```

---

## Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Conversational Client                                │
│                                                                             │
│  ┌──────────┐    ┌─────────┐    ┌───────────┐    ┌─────────────────────┐   │
│  │Microphone│───▶│  VAD    │───▶│ Segmentor │───▶│ Whisper Server (WS) │   │
│  │(sounddev)│    │(TenVAD) │    │           │    │     :8765           │   │
│  └──────────┘    └─────────┘    └───────────┘    └──────────┬──────────┘   │
│                                                              │              │
│                                                              ▼              │
│  ┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌─────────────┐       │
│  │ Speaker  │◀───│AudioPlayer  │◀───│TTS Server │◀───│ LLM Client  │       │
│  │(sounddev)│    │             │    │(WS :8766) │    │ (OpenAI API)│       │
│  └──────────┘    └─────────────┘    └───────────┘    └─────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## State Machine

```
        ┌────────────────────────────────────────┐
        │                                        │
        ▼                                        │
    ┌───────┐                              ┌─────┴──────┐
    │ IDLE  │────── user speaks ──────────▶│ PROCESSING │
    └───────┘                              └────────────┘
        ▲                                        │
        │                                        │
        └──── all pipelines complete ────────────┘
              (LLM done, TTS done, audio done)
```

### IDLE State
- Waiting for user to speak
- VAD continuously monitoring microphone
- No active LLM/TTS/audio processing

### PROCESSING State
- User speech detected and being processed
- Pipeline: Transcription → LLM → TTS → Audio playback
- Monitors for user interruption

---

## Sequence Diagram: Normal Flow (No Interruption)

```
User          Mic       VAD      Segmentor    Whisper     LLM        TTS       Audio
 │             │         │          │           │          │          │          │
 │──speaks────▶│         │          │           │          │          │          │
 │             │──frame─▶│          │           │          │          │          │
 │             │         │──prob───▶│           │          │          │          │
 │             │         │          │           │          │          │          │
 │             │    ... more frames ...        │          │          │          │
 │             │         │          │           │          │          │          │
 │──stops──────│         │          │           │          │          │          │
 │             │         │──prob───▶│           │          │          │          │
 │             │         │          │──segment─▶│          │          │          │
 │             │         │          │           │          │          │          │
 │             │         │          │           │──text───▶│          │          │
 │             │         │          │           │          │          │          │
 │             │         │          │           │          │─sent1───▶│          │
 │             │         │          │           │          │─sent2───▶│          │
 │             │         │          │           │          │          │──wav1───▶│
 │             │         │          │           │          │─sent3───▶│          │
 │◀─────────────────────────────────────────────────────────────────────plays────│
 │             │         │          │           │          │          │──wav2───▶│
 │             │         │          │           │          │          │          │
 │             │         │          │           │          │          │──wav3───▶│
 │◀──────────────────────────────────────────────────────────────────────────────│
 │             │         │          │           │          │          │          │
```

---

## Sequence Diagram: User Interruption During Playback

```
User          Mic       VAD      Main Loop    LLM        TTS       Audio
 │             │         │          │          │          │          │
 │             │         │          │          │─sent1───▶│          │
 │             │         │          │          │─sent2───▶│          │
 │             │         │          │          │          │──wav1───▶│
 │◀──────────────────────────────────────────────────────────plays───│ (chunk 1)
 │             │         │          │          │          │──wav2───▶│
 │             │         │          │          │          │          │
 │──speaks────▶│         │          │          │          │          │
 │             │──frame─▶│          │          │          │          │
 │             │         │─in_speech│          │          │          │
 │             │         │  = true  │          │          │          │
 │             │         │          │          │          │          │
 │             │         │          │◀─INTERRUPT DETECTED!            │
 │             │         │          │          │          │          │
 │             │         │          │─abort()─▶│          │          │
 │             │         │          │─────────abort()────▶│          │
 │             │         │          │────────────────abort()────────▶│
 │             │         │          │          │          │          │──fade out
 │             │         │          │          │          │          │
 │             │         │          │  [generation++]     │          │
 │             │         │          │  [keep 1 sentence   │          │
 │             │         │          │   in history]       │          │
 │             │         │          │          │          │          │
 │──continues──▶         │          │          │          │          │
 │  speaking   │         │          │          │          │          │
 │             │         │          │          │          │          │
 │──stops──────│         │          │          │          │          │
 │             │         │──segment▶│          │          │          │
 │             │         │          │──────────▶ new LLM request      │
 │             │         │          │          │          │          │
```

---

## Interrupt Detection Logic

Located in main loop (`scripts/conversational_client.py`):

```python
# Check if speech just started (edge detection)
if self.segmentor.in_speech and not self.was_in_speech:
    # Speech START detected!
    if self.state == State.PROCESSING:
        # User is interrupting - abort everything!

        # 1. Remember how many chunks finished playing
        played_count = self.audio_player.chunks_completed

        # 2. Abort all services
        self.llm_client.abort()      # Stop LLM generation
        self.tts_client.abort()      # Stop TTS, increment generation
        self.audio_player.abort()    # Fade out audio

        # 3. Keep only fully-played sentences in history
        if played_count > 0:
            played_sentences = self.current_assistant_sentences[:played_count]
            self.conversation_history.append(Message(
                role="assistant",
                content=" ".join(played_sentences)
            ))

        # 4. Reset tracking
        self.current_assistant_sentences = []
        self.audio_player.reset_tracking()

# Update edge detector
self.was_in_speech = self.segmentor.in_speech
```

---

## Generation Counter (Stale Audio Prevention)

Problem: After abort, late TTS audio from the old request could still arrive and play.

Solution: Generation counter in `TTSClient`:

```
Timeline:
─────────────────────────────────────────────────────────────────────▶

1. Stockholm request starts
   generation=0, active_generation=0

2. TTS server processing Stockholm sentences...

3. User interrupts → abort() called
   generation=1 (incremented)

4. Oskarshamn request starts
   active_generation=1 (captures current generation)

5. Late Stockholm audio arrives from server
   Check: active_generation(0) == generation(1)? NO → DISCARD

6. Oskarshamn audio arrives
   Check: active_generation(1) == generation(1)? YES → ACCEPT
```

---

## Component Abort Behavior

### LLMClient.abort()
- Sets `should_abort = True`
- Streaming loop checks flag and stops
- Clears pending sentences

### TTSClient.abort()
- Clears text queue (pending synthesis requests)
- Sends cancel messages to server for in-flight requests
- Clears ready audio buffer
- Sets `should_stop = True`
- **Increments `generation`** (invalidates late responses)

### AudioPlayer.abort()
- Clears audio queue
- Sets `should_stop = True`
- Triggers fade-out on currently playing chunk
- Does NOT increment `chunks_completed` for interrupted chunk

---

## Conversation History Management

```
Scenario: AI says 3 sentences, user interrupts during sentence 2

Sentences generated: ["Hello.", "How are you?", "Nice weather."]
Chunks played:       [  done  ] [ playing... ] [ queued ]
                                      ↑
                               USER INTERRUPTS

Result:
- chunks_completed = 1 (only "Hello." finished)
- History keeps: "Hello."
- "How are you?" and "Nice weather." are discarded
- User's interruption becomes next user message
```

---

## VAD Parameters (config.yaml)

```yaml
vad:
  threshold: 0.35      # Probability threshold to detect speech
  ema_alpha: 0.25      # Smoothing factor (lower = more smoothing)
  pre_ms: 600          # Audio to capture BEFORE VAD triggers
  hang_ms: 1200        # Silence duration before segment ends
  post_ms: 400         # Audio to capture AFTER segment ends
  min_speech_ms: 700   # Minimum segment duration (filters noise)
```

### Pre-buffer (pre_ms)
Captures audio from BEFORE the VAD detected speech. Important because:
- VAD has detection latency (EMA smoothing + min_start_frames)
- First syllables might be spoken before VAD triggers
- pre_ms=600 captures 600ms before detection point

---

## Data Flow Summary

```
1. AUDIO CAPTURE
   Microphone → audio_callback → audio_queue (thread-safe)

2. VAD PROCESSING
   Main loop drains queue → VAD.prob_speech() → Segmentor.update()

3. SPEECH SEGMENTATION
   Segmentor accumulates frames → detects speech end → emits segment

4. TRANSCRIPTION
   Segment → WebSocket → Whisper server → transcription text

5. LLM GENERATION
   Transcription → conversation history → LLM API → streaming sentences

6. TTS SYNTHESIS
   Sentences → WebSocket → TTS server → WAV audio chunks

7. AUDIO PLAYBACK
   WAV chunks → AudioPlayer queue → sounddevice OutputStream

8. INTERRUPT DETECTION (parallel)
   VAD monitors for speech start during PROCESSING state
   On detection → abort all downstream components
```
