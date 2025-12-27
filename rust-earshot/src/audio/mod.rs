//! Audio capture and playback

mod capture;
pub mod fft;
mod playback;

pub use capture::AudioCapture;
pub use capture::list_input_devices;
pub use fft::{FFTProcessor, FFTSnapshot, NUM_DISPLAY_BINS};
pub use playback::AudioPlayer;
pub use playback::list_output_devices;
