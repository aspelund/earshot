//! Speech segmentor - produces utterance segments with pre/post padding

use chrono::{DateTime, TimeZone, Utc};
use std::collections::VecDeque;
use tracing::debug;

use crate::config::VadConfig;

/// A complete speech segment
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// PCM16 audio bytes (little-endian)
    pub audio_bytes: Vec<u8>,
    /// Start time in ISO-8601 format
    pub start_iso: String,
    /// End time in ISO-8601 format
    pub end_iso: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

/// Stateful segmentor that uses VAD probabilities to produce utterance segments
/// with pre/post padding and hangover.
pub struct Segmentor {
    // Config
    sample_rate: u32,
    frame_ms: u32,
    frame_samples: usize,

    start_threshold: f32,
    end_threshold: f32,
    ema_alpha: f32,
    pre_frames: usize,
    hang_frames: usize,
    post_frames: usize,
    max_segment_frames: usize,
    min_start_frames: usize,
    min_speech_ms: u32,

    // State
    in_speech: bool,
    ring_buffer: VecDeque<Vec<i16>>,
    current_segment: Vec<Vec<i16>>,
    last_voiced_index: isize,
    segment_start_ms: i64,
    frames_since_start: usize,
    consec_voiced: usize,

    // EMA smoothed probability
    prob_ema: f32,
}

impl Segmentor {
    /// Create a new segmentor
    pub fn new(sample_rate: u32, frame_ms: u32, vad_cfg: &VadConfig) -> Self {
        let frame_samples = (sample_rate * frame_ms / 1000) as usize;

        let pre_frames = (vad_cfg.pre_ms / frame_ms).max(1) as usize;
        let post_frames = (vad_cfg.post_ms / frame_ms).max(1) as usize;
        let hang_frames = (vad_cfg.hang_ms / frame_ms).max(1) as usize;
        let max_segment_frames = ((vad_cfg.max_segment_s * 1000.0) as u32 / frame_ms).max(1) as usize;

        let ring_buffer_size = pre_frames.max(post_frames);

        debug!(
            "Segmentor: pre={}ms ({}f), hang={}ms ({}f), post={}ms ({}f), max={}s",
            vad_cfg.pre_ms, pre_frames,
            vad_cfg.hang_ms, hang_frames,
            vad_cfg.post_ms, post_frames,
            vad_cfg.max_segment_s
        );

        Self {
            sample_rate,
            frame_ms,
            frame_samples,

            start_threshold: vad_cfg.start_threshold,
            end_threshold: vad_cfg.end_threshold,
            ema_alpha: vad_cfg.ema_alpha,
            pre_frames,
            hang_frames,
            post_frames,
            max_segment_frames,
            min_start_frames: vad_cfg.min_start_frames as usize,
            min_speech_ms: vad_cfg.min_speech_ms,

            in_speech: false,
            ring_buffer: VecDeque::with_capacity(ring_buffer_size),
            current_segment: Vec::new(),
            last_voiced_index: -1,
            segment_start_ms: 0,
            frames_since_start: 0,
            consec_voiced: 0,

            prob_ema: 0.0,
        }
    }

    /// Convert milliseconds to ISO-8601 string
    fn ms_to_iso(ms: i64) -> String {
        Utc.timestamp_millis_opt(ms)
            .single()
            .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string())
            .unwrap_or_else(|| "1970-01-01T00:00:00.000Z".to_string())
    }

    /// Get current time in milliseconds
    fn now_ms() -> i64 {
        Utc::now().timestamp_millis()
    }

    /// Update segmentor with a new frame and VAD probability
    ///
    /// Returns Some(SpeechSegment) when a segment is complete, otherwise None.
    ///
    /// # Arguments
    /// * `frame_pcm16` - Audio frame as i16 PCM samples
    /// * `p_speech` - Speech probability from VAD [0, 1]
    pub fn update(&mut self, frame_pcm16: Vec<i16>, p_speech: f32) -> Option<SpeechSegment> {
        // Smooth probability with EMA
        self.prob_ema = self.ema_alpha * p_speech + (1.0 - self.ema_alpha) * self.prob_ema;

        // Feed ring buffer (used for pre/post padding)
        if self.ring_buffer.len() >= self.ring_buffer.capacity().max(self.pre_frames) {
            self.ring_buffer.pop_front();
        }
        self.ring_buffer.push_back(frame_pcm16.clone());

        // Determine voiced based on hysteresis
        let threshold = if self.in_speech {
            self.end_threshold
        } else {
            self.start_threshold
        };
        let is_voiced = self.prob_ema >= threshold;

        let now_ms_val = Self::now_ms();

        // ---------- Not currently inside a segment ----------
        if !self.in_speech {
            if is_voiced {
                self.consec_voiced += 1;
            } else {
                self.consec_voiced = 0;
            }

            if self.consec_voiced >= self.min_start_frames {
                self.in_speech = true;
                // Start time is "now minus pre_ms"
                self.segment_start_ms = now_ms_val - (self.ring_buffer.len() as i64 * self.frame_ms as i64);

                // Include pre-buffer
                let pre_start = self.ring_buffer.len().saturating_sub(self.pre_frames);
                self.current_segment = self.ring_buffer
                    .iter()
                    .skip(pre_start)
                    .cloned()
                    .collect();

                self.last_voiced_index = self.current_segment.len() as isize - 1;
                self.frames_since_start = self.current_segment.len();

                debug!(
                    "Speech started: prob_ema={:.2}, pre_frames={}",
                    self.prob_ema,
                    self.current_segment.len()
                );
            }
            return None;
        }

        // ---------- Inside a segment ----------
        self.current_segment.push(frame_pcm16);
        self.frames_since_start += 1;

        if is_voiced {
            self.last_voiced_index = self.current_segment.len() as isize - 1;
        }

        let frames_since_last_voice = (self.current_segment.len() as isize - 1) - self.last_voiced_index;

        // Force flush if too long
        if self.frames_since_start >= self.max_segment_frames {
            debug!("Flushing segment: max length reached");
            return self.flush("max_length", now_ms_val);
        }

        // End condition: no voice for hangover duration
        if frames_since_last_voice >= self.hang_frames as isize {
            debug!(
                "Flushing segment: end of speech (silence for {} frames)",
                frames_since_last_voice
            );
            return self.flush("eos", now_ms_val);
        }

        None
    }

    /// Flush the current segment
    fn flush(&mut self, reason: &str, now_ms_val: i64) -> Option<SpeechSegment> {
        // Append post padding from ring buffer if needed
        let frames_since_last_voice = (self.current_segment.len() as isize - 1) - self.last_voiced_index;
        let extra_post = (self.post_frames as isize - frames_since_last_voice).max(0) as usize;

        if extra_post > 0 {
            let tail: Vec<_> = self.ring_buffer
                .iter()
                .rev()
                .take(extra_post)
                .cloned()
                .collect();
            for frame in tail.into_iter().rev() {
                self.current_segment.push(frame);
            }
        }

        if self.current_segment.is_empty() {
            self.reset_after_flush(0.0);
            return None;
        }

        // Convert frames to bytes
        let total_samples: usize = self.current_segment.iter().map(|f| f.len()).sum();
        let mut audio_bytes = Vec::with_capacity(total_samples * 2);
        for frame in &self.current_segment {
            for &sample in frame {
                audio_bytes.extend_from_slice(&sample.to_le_bytes());
            }
        }

        // Compute times
        let start_ms_val = self.segment_start_ms;
        let duration_ms = (self.current_segment.len() as u64) * (self.frame_ms as u64);
        let end_ms_val = start_ms_val + duration_ms as i64;

        // Enforce min speech duration
        if duration_ms < self.min_speech_ms as u64 {
            debug!(
                "Dropping short segment: {} ms < {} ms",
                duration_ms, self.min_speech_ms
            );
            self.reset_after_flush(0.3);
            return None;
        }

        let start_iso = Self::ms_to_iso(start_ms_val);
        let end_iso = Self::ms_to_iso(end_ms_val);

        debug!(
            "Segment complete: {} ms, {} bytes, reason={}",
            duration_ms,
            audio_bytes.len(),
            reason
        );

        // Reset state
        self.reset_after_flush(0.3);

        Some(SpeechSegment {
            audio_bytes,
            start_iso,
            end_iso,
            duration_ms,
        })
    }

    fn reset_after_flush(&mut self, ema_decay: f32) {
        self.in_speech = false;
        self.current_segment.clear();
        self.last_voiced_index = -1;
        self.segment_start_ms = 0;
        self.frames_since_start = 0;
        self.consec_voiced = 0;

        if ema_decay > 0.0 {
            self.prob_ema *= ema_decay;
        }
    }

    /// Check if currently in speech
    pub fn is_in_speech(&self) -> bool {
        self.in_speech
    }

    /// Get current EMA probability
    pub fn prob_ema(&self) -> f32 {
        self.prob_ema
    }

    /// Force flush any pending segment
    pub fn force_flush(&mut self) -> Option<SpeechSegment> {
        if self.in_speech && !self.current_segment.is_empty() {
            self.flush("forced", Self::now_ms())
        } else {
            None
        }
    }
}
