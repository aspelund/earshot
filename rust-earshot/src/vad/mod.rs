//! Voice Activity Detection and Speech Segmentation

mod silero;
mod segmentor;

pub use silero::SileroVad;
pub use segmentor::{Segmentor, SpeechSegment};
