//! Lock-free FFT data channel for audio visualization
//!
//! Uses crossbeam's ArrayQueue for zero-lock communication between
//! the audio processing thread and the GUI thread.

use crate::audio::fft::{FFTSnapshot, NUM_DISPLAY_BINS};
use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

/// Lock-free channel for passing FFT data from audio thread to GUI
pub struct FFTChannel {
    queue: Arc<ArrayQueue<FFTSnapshot>>,
}

impl FFTChannel {
    /// Create a new FFT channel with capacity for 2 snapshots
    pub fn new() -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(2)),
        }
    }

    /// Get a sender handle for the audio thread
    pub fn sender(&self) -> FFTSender {
        FFTSender {
            queue: Arc::clone(&self.queue),
        }
    }

    /// Get a receiver handle for the GUI thread
    pub fn receiver(&self) -> FFTReceiver {
        FFTReceiver {
            queue: Arc::clone(&self.queue),
            last_snapshot: FFTSnapshot::default(),
        }
    }
}

impl Default for FFTChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Sender half of the FFT channel (used by audio thread)
#[derive(Clone)]
pub struct FFTSender {
    queue: Arc<ArrayQueue<FFTSnapshot>>,
}

impl FFTSender {
    /// Send an FFT snapshot, dropping the oldest if the queue is full
    pub fn send(&self, snapshot: FFTSnapshot) {
        // If queue is full, pop the old one and push the new one
        if self.queue.push(snapshot.clone()).is_err() {
            let _ = self.queue.pop();
            let _ = self.queue.push(snapshot);
        }
    }
}

/// Receiver half of the FFT channel (used by GUI thread)
pub struct FFTReceiver {
    queue: Arc<ArrayQueue<FFTSnapshot>>,
    last_snapshot: FFTSnapshot,
}

impl FFTReceiver {
    /// Try to receive the latest FFT snapshot
    ///
    /// Returns the newest available snapshot, or None if the queue is empty.
    /// Drains any older snapshots to always get the most recent data.
    pub fn try_recv(&mut self) -> Option<&FFTSnapshot> {
        // Drain all available snapshots and keep the last one
        let mut received = false;
        while let Some(snapshot) = self.queue.pop() {
            self.last_snapshot = snapshot;
            received = true;
        }

        if received {
            Some(&self.last_snapshot)
        } else {
            None
        }
    }

    /// Get the last received snapshot (or default if none received yet)
    pub fn last(&self) -> &FFTSnapshot {
        &self.last_snapshot
    }

    /// Get a copy of the current magnitudes for visualization
    pub fn magnitudes(&self) -> &[f32; NUM_DISPLAY_BINS] {
        &self.last_snapshot.magnitudes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_basic() {
        let channel = FFTChannel::new();
        let sender = channel.sender();
        let mut receiver = channel.receiver();

        // Initially empty
        assert!(receiver.try_recv().is_none());

        // Send a snapshot
        let mut snapshot = FFTSnapshot::default();
        snapshot.bass_energy = 0.5;
        sender.send(snapshot);

        // Should receive it
        let received = receiver.try_recv().unwrap();
        assert!((received.bass_energy - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_channel_overflow() {
        let channel = FFTChannel::new();
        let sender = channel.sender();
        let mut receiver = channel.receiver();

        // Send more than capacity
        for i in 0..5 {
            let mut snapshot = FFTSnapshot::default();
            snapshot.bass_energy = i as f32 / 10.0;
            sender.send(snapshot);
        }

        // Should get the latest
        let received = receiver.try_recv().unwrap();
        assert!(received.bass_energy >= 0.3); // Should be 0.4 (the last one sent)
    }

    #[test]
    fn test_last_snapshot() {
        let channel = FFTChannel::new();
        let sender = channel.sender();
        let mut receiver = channel.receiver();

        let mut snapshot = FFTSnapshot::default();
        snapshot.mid_energy = 0.8;
        sender.send(snapshot);

        // Receive it
        receiver.try_recv();

        // last() should return the same value
        assert!((receiver.last().mid_energy - 0.8).abs() < 0.001);

        // Even after queue is empty, last() still works
        assert!(receiver.try_recv().is_none());
        assert!((receiver.last().mid_energy - 0.8).abs() < 0.001);
    }
}
