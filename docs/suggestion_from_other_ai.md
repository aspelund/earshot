This is a sophisticated architecture! You have built a classic **turn-taking pipeline** (VAD → STT → LLM → TTS) with an added **barge-in** (interruption) mechanism.

The issue you are describing—"speaking while it's downloading audio"—is a classic race condition in voice assistants. The "Generation Counter" is a great pattern, but it often fails if the network calls are **blocking**. If your Python code is stuck waiting for a WebSocket message or an HTTP response, it cannot see that you have updated `should_abort = True` until the network call finishes. By then, it might push "zombie" audio into the queue just as you are trying to clear it.

Here is a comprehensive technical briefing you can provide to your coding agent. It analyzes the current architecture and provides specific implementation patterns to fix the state handling issues using Python's equivalent of "Signals" (`threading.Event` or `cancellation` tokens).

---

# Agent Briefing: Enhancing Interrupt Reliability in Earshot

**Objective:** Fix race conditions where user interruption fails to abort network operations immediately (specifically during LLM generation or TTS download), leading to state desync or "zombie" audio playback.

## 1. Architectural Context

* **System:** Voice-to-Voice Client (Mic → VAD → Whisper → LLM → TTS → Speaker).
* **Current State Machine:** `IDLE` vs `PROCESSING`.
* **Current Interrupt Logic:**
* VAD edge detection sets `abort()` flags on components.
* Components use boolean flags (`should_abort`) and a `generation` integer to invalidate stale data.


* **The Problem:** The boolean check relies on the loop cycling. If a component is blocked on I/O (e.g., `ws.recv()`), it misses the abort signal until the I/O completes.

## 2. The Diagnosis

The likely cause of the "speaking while downloading" bug is **Blocking I/O Latency**.
If the user interrupts while the TTS client is doing `websocket.recv()`, the thread is frozen. The Main Loop detects the interrupt and calls `tts.abort()`, which sets `self.should_stop = True`. However, the TTS thread is sleeping on the network socket. It only wakes up when the server sends the audio. It then receives the audio, checks the flag (maybe too late), or inadvertently queues the audio before the logic catches up.

## 3. Implementation Strategy (The "Fix")

To replicate the behavior of TypeScript `AbortSignal` in Python, we need to move from **Passive Flag Checking** to **Active Cancellation**.

### A. Replace Booleans with `threading.Event`

Instead of `self.should_stop = False`, use `self.cancel_event = threading.Event()`.

* **Why:** It is thread-safe and allows for timeout-based waiting.

### B. Non-Blocking Network Loops

Modify the `WebSocketTTSClient` and `LLMClient` to stop blocking indefinitely on network sockets.

**Current (Likely) Logic:**

```python
# Problematic: Blocks until server replies, ignoring interrupts
message = websocket.recv() 
if self.should_stop: return # Too late!

```

**Proposed Logic (using select/timeout):**

```python
# Better: Polls socket while checking for cancellation
while not self.cancel_event.is_set():
    try:
        # Wait for data with a short timeout (e.g., 0.1s) to allow checking the event
        message = websocket.recv(timeout=0.1)
        process_audio(message)
    except TimeoutError:
        continue # Loop back and check cancel_event again

```

### C. The "Check-Then-Act" Pattern (Double-Check Locking)

We must check the generation counter *atomically* before any side effect (like adding to a queue).

**Refined Logic for `WebSocketTTSClient`:**

```python
def on_audio_received(self, audio_chunk, generation_at_request_time):
    # CRITICAL: Check generation match immediately before queueing
    # If the global generation bumped up while we were downloading, DISCARD.
    if generation_at_request_time != self.current_generation:
        print(f"Discarding stale audio (Gen {generation_at_request_time} vs {self.current_generation})")
        return

    # Check explicit cancel flag
    if self.cancel_event.is_set():
        return

    self.audio_queue.put(audio_chunk)

```

### D. Hard Reset on Interrupt

When the Main Loop detects an interrupt, it shouldn't just set a flag; it needs to aggressively clear the pipelines.

**Update `conversational_client.py` Interrupt Block:**

```python
if self.segmentor.in_speech and not self.was_in_speech:
    if self.state == State.PROCESSING:
        # 1. Bump Generation (Invalidates all in-flight packets instantly)
        self.generation_id += 1 
        
        # 2. Trigger Events (Wakes up any sleeping threads if using wait())
        self.tts_client.cancel_event.set()
        self.llm_client.cancel_event.set()
        
        # 3. Purge Queues (Remove data that beat the race condition)
        self.audio_player.clear_queue() 
        self.tts_client.clear_pending()
        
        # 4. Send Server-Side Cancellation (If supported by API)
        # e.g., self.tts_client.send_json({"type": "cancel"})

```

## 4. Summary of Tasks for the Agent

1. **Refactor `should_stop` booleans** to `threading.Event` objects named `cancel_event`.
2. **Add Timeouts** to all `socket.recv()` or `queue.get()` calls in the threads so they wake up periodically to check `cancel_event.is_set()`.
3. **Strict Generation Validation:** Ensure `generation_id` is captured at the *start* of a request and checked *after* every network receive operation before processing the data.
4. **Queue Purging:** Ensure `abort()` explicitly empties the `audio_queue` *after* setting the cancel event, to catch any packets that slipped in during the exact millisecond of the interrupt.

---

**Would you like me to create a specific prompt for the coding agent that includes the relevant code snippets for the `WebSocketTTSClient` modification?**