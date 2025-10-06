"""
Tests for UTC timestamp utilities.
"""
import pytest
from src.utils_time import utc_now_iso, now_ms
from datetime import datetime, timezone


def test_utc_now_iso_format():
    """Test ISO format with Z suffix."""
    iso = utc_now_iso()
    assert iso.endswith("Z")
    assert "T" in iso
    # Should parse without error
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    assert dt.tzinfo == timezone.utc


def test_now_ms_is_reasonable():
    """Test milliseconds since epoch is in reasonable range."""
    ms = now_ms()
    # Should be after 2020 and before 2100
    assert ms > 1577836800000  # Jan 1, 2020
    assert ms < 4102444800000  # Jan 1, 2100


def test_timestamps_are_close():
    """Test that ISO and ms timestamps are consistent."""
    iso = utc_now_iso()
    ms = now_ms()

    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    iso_ms = int(dt.timestamp() * 1000)

    # Should be within 100ms of each other
    assert abs(iso_ms - ms) < 100
