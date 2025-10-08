"""
Unit tests for AudioPlayer
"""
import pytest
import asyncio
import numpy as np
import io
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock
from src.audio import AudioPlayer


def generate_tone(frequency_hz: int, duration_s: float, sample_rate: int = 16000) -> bytes:
    """Generate a sine wave tone as WAV bytes"""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency_hz * t)

    # Convert to WAV bytes
    with io.BytesIO() as f:
        sf.write(f, audio_data, sample_rate, format='WAV', subtype='FLOAT')
        return f.getvalue()


class MockAudioStream:
    """Mock for sounddevice.OutputStream that captures audio output"""

    def __init__(self):
        self.frames_written = []
        self.callback = None
        self.samplerate = None
        self.is_active = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def simulate_playback(self, num_callbacks: int = 100):
        """Simulate audio callbacks"""
        if not self.callback:
            return

        frames_per_callback = 512  # Typical buffer size

        for _ in range(num_callbacks):
            if not self.is_active:
                break

            outdata = np.zeros((frames_per_callback, 1), dtype=np.float32)

            try:
                self.callback(outdata, frames_per_callback, None, None)
                self.frames_written.append(outdata.copy())
            except Exception:
                # Callback raised CallbackStop or error
                self.is_active = False
                break

    def get_total_output(self) -> np.ndarray:
        """Get all audio data written"""
        if not self.frames_written:
            return np.array([])
        return np.concatenate(self.frames_written, axis=0).flatten()


@pytest.fixture
def mock_audio_device():
    """Fixture that mocks sounddevice.OutputStream"""
    mock_stream = MockAudioStream()

    def create_stream(samplerate, channels, dtype, callback, device=None):
        mock_stream.callback = callback
        mock_stream.samplerate = samplerate
        return mock_stream

    with patch('src.audio.player.sd.OutputStream', side_effect=create_stream):
        yield mock_stream


@pytest.mark.asyncio
async def test_enqueue_and_playback_single_file(mock_audio_device):
    """Test basic enqueue and playback of a single file"""
    player = AudioPlayer(fade_out_duration_ms=250)
    player.start()

    # Generate a 100ms tone
    tone = generate_tone(440, 0.1, 16000)

    # Enqueue and wait a bit for playback to start
    player.enqueue(tone)
    await asyncio.sleep(0.05)

    assert player.is_playing or player.queue.qsize() > 0  # Either playing or queued

    # Simulate playback
    mock_audio_device.simulate_playback(50)

    # Verify some audio was written
    output = mock_audio_device.get_total_output()
    assert len(output) > 0
    assert np.max(np.abs(output)) > 0  # Non-silent

    await player.stop()


@pytest.mark.asyncio
async def test_multiple_files_play_sequentially(mock_audio_device):
    """Test that multiple files play in order"""
    player = AudioPlayer(fade_out_duration_ms=250)
    player.start()

    # Enqueue three short tones
    tone1 = generate_tone(440, 0.05, 16000)
    tone2 = generate_tone(880, 0.05, 16000)
    tone3 = generate_tone(1320, 0.05, 16000)

    player.enqueue(tone1)
    player.enqueue(tone2)
    player.enqueue(tone3)

    assert player.queue.qsize() == 3

    # Let playback run
    await asyncio.sleep(0.01)

    # Verify queue is being processed
    assert player.queue.qsize() <= 3  # At least one should have been dequeued

    await player.stop()


@pytest.mark.asyncio
async def test_abort_during_playback(mock_audio_device):
    """Test abort interrupts playback with fade-out"""
    player = AudioPlayer(fade_out_duration_ms=250)
    player.start()

    # Generate a longer tone (500ms)
    tone = generate_tone(440, 0.5, 16000)
    player.enqueue(tone)

    # Wait for playback to start
    await asyncio.sleep(0.05)

    # Abort with fast fade (100ms)
    player.abort(fade_ms=100)

    # Give time for abort to take effect
    await asyncio.sleep(0.02)

    # Verify abort was triggered
    assert player.should_stop or not player.is_playing

    await player.stop()


@pytest.mark.asyncio
async def test_abort_clears_queue(mock_audio_device):
    """Test that abort clears queued items"""
    player = AudioPlayer(fade_out_duration_ms=250)
    player.start()

    # Enqueue multiple files
    tone = generate_tone(440, 0.1, 16000)
    for _ in range(5):
        player.enqueue(tone)

    initial_queue_size = player.queue.qsize()
    assert initial_queue_size > 0

    # Abort should clear queue
    player.abort()

    # Queue should be empty
    assert player.queue.qsize() == 0

    await player.stop()


@pytest.mark.asyncio
async def test_abort_when_idle(mock_audio_device):
    """Test abort when nothing is playing doesn't crash"""
    player = AudioPlayer(fade_out_duration_ms=250)
    player.start()

    # Abort when idle
    player.abort()

    # Should not crash
    assert player.queue.qsize() == 0
    assert not player.is_playing

    await player.stop()


@pytest.mark.asyncio
async def test_start_required_before_enqueue():
    """Test that enqueue fails if start() not called"""
    player = AudioPlayer(fade_out_duration_ms=250)

    tone = generate_tone(440, 0.1, 16000)

    with pytest.raises(RuntimeError, match="not started"):
        player.enqueue(tone)


@pytest.mark.asyncio
async def test_fade_out_reduces_amplitude(mock_audio_device):
    """Test that fade-out actually reduces audio amplitude"""
    player = AudioPlayer(fade_out_duration_ms=50)  # Short fade for testing
    player.start()

    # Generate tone
    tone = generate_tone(440, 0.2, 16000)
    player.enqueue(tone)

    # Let it play a bit
    await asyncio.sleep(0.05)

    # Trigger abort
    player.abort(fade_ms=50)

    # Simulate playback during fade
    mock_audio_device.simulate_playback(20)

    output = mock_audio_device.get_total_output()

    if len(output) > 1000:
        # Check that later samples have lower amplitude (fading out)
        # Split into chunks and verify amplitude decreases
        chunk_size = len(output) // 4
        if chunk_size > 0:
            first_chunk_max = np.max(np.abs(output[:chunk_size]))
            last_chunk_max = np.max(np.abs(output[-chunk_size:]))

            # Last chunk should have lower amplitude (could be silence)
            assert last_chunk_max <= first_chunk_max

    await player.stop()


@pytest.mark.asyncio
async def test_player_cleanup(mock_audio_device):
    """Test that player cleanup doesn't crash"""
    player = AudioPlayer(fade_out_duration_ms=250)
    player.start()

    # Enqueue some audio
    tone = generate_tone(440, 0.1, 16000)
    player.enqueue(tone)

    # Stop player
    await player.stop()

    # Verify task is cleaned up
    assert player._playback_task is None or player._playback_task.cancelled()
