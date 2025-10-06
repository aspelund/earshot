# Fix Proposal: WebSocket Audio Streaming Reliability

**Issue:** Audio frames are being silently dropped during WebSocket streaming, causing transcription gaps and poor quality.

**Date:** 2025-10-06
**Status:** Proposed

---

## Problem Analysis

### Root Cause

The client-side audio streaming has a critical bottleneck:

**Location:** `scripts/stream_client.py:68`
```python
self.audio_queue = asyncio.Queue(maxsize=100)
```

**Issues:**
1. **Queue too small** - 100 frames × 30ms = 3 seconds buffer
2. **Silent frame drops** - Line 33-35: `except: pass` drops frames without logging
3. **Network latency sensitivity** - If WebSocket send takes >30ms, queue fills up
4. **No batching** - Each 30ms frame = separate WebSocket message (network overhead)

### Impact

```
Audio capture rate:    33.3 frames/sec (30ms each)
Network send rate:     Variable (depends on latency)

If network latency > 30ms per message:
  → Queue fills in ~3 seconds
  → Frames start dropping
  → Audio has gaps
  → VAD misses speech
  → Transcriptions incomplete
```

**Real-world scenario:**
- 50ms network latency (typical WiFi)
- Client captures frame every 30ms
- Client can only send every 50ms
- Queue fills: 100 frames ÷ (50ms/30ms backlog rate) = **~1.5 minutes to failure**
- Once full, drops 40% of frames (20ms out of every 50ms)

---

## Proposed Solutions

### Option 1: Quick Fix (Increase Buffer + Add Logging) ⭐ **Recommended**

**Changes:**
1. Increase queue size: `maxsize=100` → `maxsize=1000` (30 seconds buffer)
2. Add drop counter and periodic warnings
3. Add queue fullness metrics

**Pros:**
- Minimal code change
- Immediate improvement
- Handles network jitter (up to 30s of accumulated delay)

**Cons:**
- Still drops frames under sustained slow network
- Doesn't address root cause (blocking sends)

**Implementation:**
```python
# In stream_client.py
self.audio_queue = asyncio.Queue(maxsize=1000)  # 30s buffer
self.dropped_frames = 0
self.total_frames = 0

# In audio_callback:
try:
    self.audio_queue.put_nowait(pcm16.tobytes())
    self.total_frames += 1
except asyncio.QueueFull:
    self.dropped_frames += 1
    if self.dropped_frames % 100 == 0:
        drop_rate = self.dropped_frames / self.total_frames * 100
        print(f"⚠ Warning: Dropped {self.dropped_frames} frames ({drop_rate:.1f}%)",
              file=sys.stderr)
```

---

### Option 2: Frame Batching (Reduce Network Overhead)

**Changes:**
1. Batch N frames (e.g., 10 frames = 300ms) into single WebSocket message
2. Reduce send frequency from 33 msg/sec to 3 msg/sec
3. Server-side: Split batched frames back into individual chunks

**Pros:**
- 10x reduction in WebSocket messages
- Amortizes network latency over multiple frames
- More efficient network usage
- Still maintains 300ms granularity

**Cons:**
- Adds 300ms buffering delay
- Requires server-side changes
- More complex implementation

**Implementation:**
```python
# Client side - batch frames
async def stream_audio(self):
    batch = []
    batch_size = 10  # 10 frames = 300ms

    while True:
        audio_bytes = await self.audio_queue.get()
        batch.append(audio_bytes)

        if len(batch) >= batch_size:
            # Send batched frames as one message
            batched = b''.join(batch)
            await ws.send(batched)
            batch = []

# Server side - split batches
def push_audio(self, pcm16_bytes: bytes):
    # Split into frame_samples chunks
    frame_size = self.frame_samples * 2  # 2 bytes per int16
    for i in range(0, len(pcm16_bytes), frame_size):
        chunk = pcm16_bytes[i:i+frame_size]
        if len(chunk) == frame_size:
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            self.queue.put(pcm16)
```

---

### Option 3: Adaptive Buffering (Dynamic Queue Management)

**Changes:**
1. Monitor queue fullness continuously
2. Dynamically adjust buffer size based on network conditions
3. Warning levels: 50% (info), 75% (warning), 90% (error)
4. Optional: Pause microphone if queue critically full

**Pros:**
- Self-tuning to network conditions
- Early warnings before drops occur
- Can prevent drops entirely by pausing capture

**Cons:**
- Most complex implementation
- Pausing capture = audio gaps anyway
- Better to let server handle backpressure

---

### Option 4: Server-Side Backpressure (Flow Control)

**Changes:**
1. Server sends acknowledgment messages
2. Client tracks in-flight frames
3. Client pauses/slows when server can't keep up
4. Implements TCP-like flow control over WebSocket

**Pros:**
- Prevents drops by matching client rate to server capacity
- Graceful degradation under load
- No silent failures

**Cons:**
- Complex protocol changes
- Adds round-trip latency
- May cause audio gaps during backpressure

---

## Recommended Implementation Plan

### Phase 1: Immediate (Quick Wins)

**Priority: HIGH**

1. **Increase client queue size** to 1000 frames (30s buffer)
2. **Add drop counting and warnings** to detect issues
3. **Add queue metrics** (current size, fullness %)

**Files to modify:**
- `scripts/stream_client.py`

**Testing:**
- Test with high network latency (50-100ms)
- Monitor drop rates over 5-10 minute sessions
- Verify no drops with 1000-frame buffer

---

### Phase 2: Medium-term (Efficiency)

**Priority: MEDIUM**

1. **Implement frame batching** (10 frames = 300ms)
2. **Update server to handle batched frames**
3. **Add network stats** (latency, bandwidth, drops)

**Files to modify:**
- `scripts/stream_client.py` (batching)
- `src/audio_network.py` (batch splitting)
- `src/stream_server.py` (stats tracking)

**Testing:**
- Verify transcription quality unchanged
- Measure network bandwidth reduction
- Test with poor network conditions

---

### Phase 3: Long-term (Robustness)

**Priority: LOW**

1. **Add adaptive buffering** with dynamic queue sizing
2. **Implement flow control** protocol
3. **Add client-side audio compression** (optional)

**Files to modify:**
- Major refactor of streaming architecture

---

## Monitoring & Debugging

### Metrics to Add

**Client side:**
```python
- audio_frames_captured (counter)
- audio_frames_dropped (counter)
- audio_queue_size (gauge)
- audio_queue_fullness_pct (gauge)
- ws_send_latency_ms (histogram)
- ws_send_errors (counter)
```

**Server side:**
```python
- audio_frames_received (counter)
- audio_frames_processed (counter)
- audio_receive_rate (gauge, frames/sec)
- ws_active_connections (gauge)
- transcription_lag_ms (histogram)
```

### Debug Mode

Add `--debug` flag to client:
```bash
python scripts/stream_client.py --server ws://... --debug
```

Shows:
- Real-time queue stats
- Frame drop warnings
- Network latency per send
- Bytes sent/received

---

## Testing Strategy

### Test Cases

1. **Normal network** (10ms latency)
   - Expected: 0% drops, queue < 10% full

2. **High latency** (100ms latency)
   - Expected: 0% drops with 1000-frame queue

3. **Packet loss** (5% loss)
   - Expected: Reconnect, resume streaming

4. **Sustained slow network** (200ms latency for 5 minutes)
   - Expected: Queue fills but no drops, some transcription delay

5. **Network interruption** (10 second disconnect)
   - Expected: Graceful reconnect, minimal audio loss

### Load Testing

```bash
# Stress test: Run for 1 hour
python scripts/stream_client.py --server ws://... &
# Monitor metrics every 5 seconds
watch -n 5 "grep -E 'dropped|queue' /tmp/client.log"
```

---

## Migration Path

### For Existing Deployments

1. **Backward compatible** - server already handles any frame rate
2. **No config changes** - improvements are internal
3. **Optional debug flags** - add `--debug` for troubleshooting

### Rollout Plan

1. **Deploy Phase 1** (queue size + logging) immediately
2. **Monitor for 1 week** - gather metrics
3. **Deploy Phase 2** (batching) if drops still occur
4. **Phase 3** only if needed for poor networks

---

## Alternative Approaches Considered

### 1. UDP Streaming
- **Pro:** Lower latency, no head-of-line blocking
- **Con:** Packet loss, requires custom protocol, no WebSocket

### 2. Audio Compression (Opus codec)
- **Pro:** 10x bandwidth reduction
- **Con:** Requires encoding/decoding, CPU overhead, complexity

### 3. Client-side VAD (only send speech)
- **Pro:** Massive bandwidth reduction
- **Con:** Misses some speech, double VAD processing, lag

### 4. Multiple WebSocket connections
- **Pro:** Parallel streams, redundancy
- **Con:** Complex, wastes bandwidth, server overhead

---

## Success Metrics

### Before Fix
- Drop rate: **Unknown** (silent drops)
- Buffer exhaustion: ~3 seconds under load
- User complaint: "Skipping long parts"

### After Phase 1
- Drop rate: **< 0.1%** (logged)
- Buffer capacity: 30 seconds
- User experience: Minimal gaps

### After Phase 2
- Network overhead: **90% reduction** (batching)
- Drop rate: **< 0.01%**
- Latency: +300ms buffering (acceptable)

---

## References

- WebSocket best practices: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API
- Asyncio queue patterns: https://docs.python.org/3/library/asyncio-queue.html
- Real-time audio streaming: https://webrtc.org/getting-started/media-devices

---

## Appendix: Code Snippets

### Client-Side: Enhanced Monitoring

```python
class AudioStreamClient:
    def __init__(self, ...):
        # ... existing code ...
        self.stats = {
            'captured': 0,
            'dropped': 0,
            'sent': 0,
            'queue_max_seen': 0
        }
        self.last_stats_print = time.time()

    def audio_callback(self, indata, frames, time_info, status):
        # ... existing conversion ...
        self.stats['captured'] += 1

        try:
            self.audio_queue.put_nowait(pcm16.tobytes())

            # Track queue fullness
            qsize = self.audio_queue.qsize()
            if qsize > self.stats['queue_max_seen']:
                self.stats['queue_max_seen'] = qsize
        except asyncio.QueueFull:
            self.stats['dropped'] += 1

    async def stream_audio(self):
        # ... existing connection code ...

        while True:
            audio_bytes = await self.audio_queue.get()
            await ws.send(audio_bytes)
            self.stats['sent'] += 1

            # Print stats every 10 seconds
            now = time.time()
            if now - self.last_stats_print > 10:
                self._print_stats()
                self.last_stats_print = now

    def _print_stats(self):
        drop_rate = (self.stats['dropped'] / max(1, self.stats['captured'])) * 100
        qmax = self.stats['queue_max_seen']
        qpct = (qmax / 1000) * 100  # Assuming maxsize=1000

        print(f"\n📊 Stats: Captured={self.stats['captured']}, "
              f"Sent={self.stats['sent']}, "
              f"Dropped={self.stats['dropped']} ({drop_rate:.2f}%), "
              f"Max Queue={qmax}/{1000} ({qpct:.0f}%)",
              file=sys.stderr)
```

---

## Implementation Checklist

### Phase 1 (Immediate)
- [ ] Increase `audio_queue` maxsize to 1000
- [ ] Add drop counter in audio_callback
- [ ] Add periodic stats logging
- [ ] Add queue fullness tracking
- [ ] Test with 50ms artificial latency
- [ ] Deploy to staging
- [ ] Monitor for 24 hours
- [ ] Deploy to production

### Phase 2 (If Needed)
- [ ] Implement frame batching in client
- [ ] Update server to split batches
- [ ] Add batch size configuration
- [ ] Test latency impact
- [ ] Benchmark network usage
- [ ] Update documentation

### Phase 3 (Optional)
- [ ] Design flow control protocol
- [ ] Implement adaptive buffering
- [ ] Add compression support
- [ ] Performance testing
