//! Thread-safe shared state for GUI communication

use super::fft_data::{FFTChannel, FFTReceiver, FFTSender};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

/// Atomic f32 wrapper for lock-free audio levels
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(val: f32) -> Self {
        Self(AtomicU32::new(val.to_bits()))
    }

    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub fn store(&self, val: f32) {
        self.0.store(val.to_bits(), Ordering::Relaxed);
    }
}

/// Pipeline state as seen by GUI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PipelineState {
    Idle = 0,
    Listening = 1,
    Processing = 2,
    Speaking = 3,
    Stopped = 4,
}

impl From<u8> for PipelineState {
    fn from(val: u8) -> Self {
        match val {
            0 => Self::Idle,
            1 => Self::Listening,
            2 => Self::Processing,
            3 => Self::Speaking,
            4 => Self::Stopped,
            _ => Self::Idle,
        }
    }
}

/// Commands from GUI to pipeline
#[derive(Debug, Clone)]
pub enum GuiCommand {
    StartListening,
    StopListening,
    Shutdown,
}

/// Shared state between GUI and pipeline
pub struct GuiState {
    /// Input audio RMS level (0.0 to 1.0)
    pub input_level: AtomicF32,
    /// Output audio RMS level (0.0 to 1.0)
    pub output_level: AtomicF32,
    /// VAD probability (0.0 to 1.0)
    pub vad_probability: AtomicF32,
    /// Current pipeline state
    state: AtomicU8,
    /// Command sender (GUI -> Pipeline)
    pub command_tx: Sender<GuiCommand>,
    /// Command receiver (Pipeline side)
    command_rx: Receiver<GuiCommand>,
    /// FFT sender (Pipeline -> GUI) - clone this for the audio thread
    fft_sender: FFTSender,
    /// FFT receiver (GUI side) - stored here but moved to app on creation
    /// Wrapped in Mutex so it can be taken even with multiple Arc references
    fft_receiver: Mutex<Option<FFTReceiver>>,
}

impl GuiState {
    pub fn new() -> Arc<Self> {
        let (command_tx, command_rx) = bounded(16);
        let fft_channel = FFTChannel::new();
        Arc::new(Self {
            input_level: AtomicF32::new(0.0),
            output_level: AtomicF32::new(0.0),
            vad_probability: AtomicF32::new(0.0),
            state: AtomicU8::new(PipelineState::Idle as u8),
            command_tx,
            command_rx,
            fft_sender: fft_channel.sender(),
            fft_receiver: Mutex::new(Some(fft_channel.receiver())),
        })
    }

    /// Get the FFT sender for the audio pipeline to use
    pub fn fft_sender(&self) -> FFTSender {
        self.fft_sender.clone()
    }

    /// Take the FFT receiver (can only be called once, typically by the app)
    pub fn take_fft_receiver(&self) -> Option<FFTReceiver> {
        self.fft_receiver.lock().ok().and_then(|mut guard| guard.take())
    }

    pub fn state(&self) -> PipelineState {
        PipelineState::from(self.state.load(Ordering::Relaxed))
    }

    pub fn set_state(&self, state: PipelineState) {
        self.state.store(state as u8, Ordering::Relaxed);
    }

    /// Try to receive a command (non-blocking)
    pub fn try_recv_command(&self) -> Option<GuiCommand> {
        self.command_rx.try_recv().ok()
    }
}

impl Default for GuiState {
    fn default() -> Self {
        let (command_tx, command_rx) = bounded(16);
        let fft_channel = FFTChannel::new();
        Self {
            input_level: AtomicF32::new(0.0),
            output_level: AtomicF32::new(0.0),
            vad_probability: AtomicF32::new(0.0),
            state: AtomicU8::new(PipelineState::Idle as u8),
            command_tx,
            command_rx,
            fft_sender: fft_channel.sender(),
            fft_receiver: Mutex::new(Some(fft_channel.receiver())),
        }
    }
}
