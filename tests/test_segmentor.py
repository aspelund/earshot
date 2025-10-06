"""
Tests for the Segmentor state machine.
"""
import pytest
import numpy as np
from src.segmentor import Segmentor


@pytest.fixture
def basic_config():
    return {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "frame_ms": 20
        },
        "vad": {
            "threshold": 0.5,
            "ema_alpha": 0.3,
            "pre_ms": 200,
            "hang_ms": 250,
            "post_ms": 200,
            "max_segment_s": 20
        }
    }


def test_frame_samples_calculation(basic_config):
    """Test that frame samples are calculated correctly."""
    seg = Segmentor(basic_config)
    # 16000 Hz * 20ms = 320 samples
    assert seg.frame_samples == 320


def test_no_flush_on_silence(basic_config):
    """Test that silence doesn't trigger segment flush."""
    seg = Segmentor(basic_config)
    frame = np.zeros(320, dtype=np.int16)

    for _ in range(100):
        result = seg.update(frame, p_speech=0.1)
        assert result is None


def test_speech_detection_triggers_segment(basic_config):
    """Test that speech probability above threshold starts segment."""
    seg = Segmentor(basic_config)
    frame = np.random.randint(-1000, 1000, 320, dtype=np.int16)

    # Feed speech frames
    for _ in range(5):
        seg.update(frame, p_speech=0.8)

    assert seg.in_speech is True


def test_hangover_triggers_flush(basic_config):
    """Test that hangover period triggers segment flush."""
    seg = Segmentor(basic_config)
    frame = np.random.randint(-1000, 1000, 320, dtype=np.int16)

    # Start speech
    for _ in range(10):
        result = seg.update(frame, p_speech=0.8)
        assert result is None

    # Hangover period (250ms = ~12 frames at 20ms)
    for i in range(15):
        result = seg.update(frame, p_speech=0.1)
        if result is not None:
            # Flush happened
            pcm_bytes, start_iso, end_iso = result
            assert isinstance(pcm_bytes, bytes)
            assert len(pcm_bytes) > 0
            assert isinstance(start_iso, str)
            assert isinstance(end_iso, str)
            break
    else:
        pytest.fail("Expected flush during hangover period")


def test_max_segment_length_flush(basic_config):
    """Test that max segment length triggers flush."""
    cfg = basic_config.copy()
    cfg["vad"]["max_segment_s"] = 1  # 1 second max
    seg = Segmentor(cfg)
    frame = np.random.randint(-1000, 1000, 320, dtype=np.int16)

    # Feed continuous speech beyond max length
    # 1 second = 50 frames at 20ms
    for i in range(60):
        result = seg.update(frame, p_speech=0.8)
        if result is not None:
            # Max length flush happened
            assert i >= 50  # Should flush around 50 frames
            break
    else:
        pytest.fail("Expected flush due to max segment length")


def test_ema_smoothing(basic_config):
    """Test that EMA smoothing affects detection."""
    seg = Segmentor(basic_config)
    frame = np.zeros(320, dtype=np.int16)

    # Single high-prob frame shouldn't immediately trigger with EMA
    seg.update(frame, p_speech=0.9)
    # EMA will be 0.3 * 0.9 + 0.7 * 0 = 0.27, below threshold of 0.5
    assert seg.prob_ema < seg.threshold

    # Multiple frames should push EMA above threshold
    for _ in range(5):
        seg.update(frame, p_speech=0.9)

    assert seg.prob_ema > seg.threshold
