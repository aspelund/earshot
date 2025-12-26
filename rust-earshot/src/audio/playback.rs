//! Audio playback with fade-out support for interruption

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

const FADE_OUT_MS: u32 = 250;
const AMPLITUDE_WINDOW: usize = 512; // Samples for RMS calculation

/// Shared state for audio playback (must be Send + Sync)
struct PlaybackState {
    queue: VecDeque<Vec<f32>>,
    current_chunk: Vec<f32>,
    position: usize,
    fade_out: bool,
    fade_samples_remaining: usize,
}

/// Thread-safe audio player handle
/// The actual cpal::Stream is kept in a separate non-Send wrapper
pub struct AudioPlayer {
    state: Arc<Mutex<PlaybackState>>,
    is_playing: Arc<AtomicBool>,
    chunks_completed: Arc<AtomicUsize>,
    current_amplitude: Arc<AtomicU32>,
    sample_rate: u32,
    device_rate: u32,
    // Keep stream alive but don't expose it (not Send)
    _stream_holder: Arc<StreamHolder>,
}

/// Wrapper to hold the cpal::Stream (not Send)
struct StreamHolder {
    stream: Mutex<Option<cpal::Stream>>,
}

// StreamHolder is not Send/Sync by default because Stream isn't
// But we never actually access the stream from other threads
// We just need to keep it alive
unsafe impl Send for StreamHolder {}
unsafe impl Sync for StreamHolder {}

impl AudioPlayer {
    /// Create a new audio player
    pub fn new(sample_rate: u32) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("No output device available"))?;

        info!("Output device: {}", device.name().unwrap_or_default());

        let target_rate = sample_rate;

        // Try to use the requested config first, fall back to supported rate
        let mut config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(target_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Query supported configurations and find best match
        let (device_rate, device_channels) = match device.supported_output_configs() {
            Ok(supported) => {
                let supported: Vec<_> = supported.collect();

                // Debug: log supported configs
                for cfg in &supported {
                    debug!("Supported output: {}ch, {}-{}Hz",
                        cfg.channels(), cfg.min_sample_rate().0, cfg.max_sample_rate().0);
                }

                // Try to find exact match first
                let has_exact = supported.iter().any(|c| {
                    c.channels() == 1 &&
                    c.min_sample_rate().0 <= target_rate &&
                    c.max_sample_rate().0 >= target_rate
                });

                if has_exact {
                    info!("Output device supports {}Hz mono natively", target_rate);
                    (target_rate, 1u16)
                } else {
                    // Find a supported config (prefer 48kHz, allow stereo)
                    let preferred_rates = [48000u32, 44100, 96000];
                    let mut found: Option<(u32, u16)> = None;

                    for rate in preferred_rates {
                        // Try mono first
                        if supported.iter().any(|c| {
                            c.channels() == 1 &&
                            c.min_sample_rate().0 <= rate &&
                            c.max_sample_rate().0 >= rate
                        }) {
                            found = Some((rate, 1));
                            break;
                        }
                        // Try stereo
                        if supported.iter().any(|c| {
                            c.channels() == 2 &&
                            c.min_sample_rate().0 <= rate &&
                            c.max_sample_rate().0 >= rate
                        }) {
                            found = Some((rate, 2));
                            break;
                        }
                    }

                    if let Some((dev_rate, channels)) = found {
                        info!("Output device uses {}Hz {}ch, will upsample from {}Hz mono",
                            dev_rate, channels, target_rate);
                        (dev_rate, channels)
                    } else if let Ok(default) = device.default_output_config() {
                        let dev_rate = default.sample_rate().0;
                        let channels = default.channels();
                        info!("Using device default {}Hz {}ch, will upsample from {}Hz",
                            dev_rate, channels, target_rate);
                        (dev_rate, channels)
                    } else {
                        (target_rate, 1)
                    }
                }
            }
            Err(e) => {
                warn!("Could not query output configs: {}, trying requested", e);
                (target_rate, 1)
            }
        };

        config.sample_rate = cpal::SampleRate(device_rate);
        config.channels = device_channels;

        let state = Arc::new(Mutex::new(PlaybackState {
            queue: VecDeque::new(),
            current_chunk: Vec::new(),
            position: 0,
            fade_out: false,
            fade_samples_remaining: 0,
        }));

        let is_playing = Arc::new(AtomicBool::new(false));
        let chunks_completed = Arc::new(AtomicUsize::new(0));
        let current_amplitude = Arc::new(AtomicU32::new(0));

        let state_clone = state.clone();
        let is_playing_clone = is_playing.clone();
        let chunks_completed_clone = chunks_completed.clone();
        let current_amplitude_clone = current_amplitude.clone();
        let fade_samples = (device_rate * FADE_OUT_MS / 1000) as usize;
        let channels = device_channels as usize;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut state = state_clone.lock().unwrap();
                let mut sum_sq = 0.0f32;
                let mut sample_count = 0usize;

                // Process in frames (one sample per channel)
                for frame in data.chunks_mut(channels) {
                    let sample_value = if state.fade_out && state.fade_samples_remaining > 0 {
                        // Fade out current audio
                        let fade_factor = state.fade_samples_remaining as f32 / fade_samples as f32;
                        let val = if state.position < state.current_chunk.len() {
                            let s = state.current_chunk[state.position] * fade_factor;
                            state.position += 1;
                            s
                        } else {
                            0.0
                        };
                        state.fade_samples_remaining -= 1;

                        if state.fade_samples_remaining == 0 {
                            // Fade complete, clear everything
                            state.queue.clear();
                            state.current_chunk.clear();
                            state.position = 0;
                            is_playing_clone.store(false, Ordering::SeqCst);
                            state.fade_out = false;
                        }
                        val
                    } else if state.position < state.current_chunk.len() {
                        // Play current chunk
                        let val = state.current_chunk[state.position];
                        state.position += 1;
                        val
                    } else if let Some(next_chunk) = state.queue.pop_front() {
                        // Move to next chunk
                        chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                        state.current_chunk = next_chunk;
                        state.position = 0;
                        if !state.current_chunk.is_empty() {
                            let val = state.current_chunk[state.position];
                            state.position += 1;
                            val
                        } else {
                            0.0
                        }
                    } else {
                        // Nothing to play
                        if is_playing_clone.load(Ordering::SeqCst) {
                            state.current_chunk.clear();
                            state.position = 0;
                            is_playing_clone.store(false, Ordering::SeqCst);
                        }
                        0.0
                    };

                    // Write sample to all channels (mono to stereo duplication)
                    for ch_sample in frame.iter_mut() {
                        *ch_sample = sample_value;
                    }

                    // Accumulate for RMS calculation
                    sum_sq += sample_value * sample_value;
                    sample_count += 1;
                }

                // Calculate and store RMS amplitude (scaled for display)
                if sample_count > 0 {
                    let rms = (sum_sq / sample_count as f32).sqrt();
                    // Scale for visibility (reduced to prevent clipping)
                    let scaled = (rms * 3.3).min(1.0);
                    current_amplitude_clone.store(scaled.to_bits(), Ordering::Relaxed);
                }
            },
            |err| warn!("Audio playback error: {}", err),
            None,
        )?;

        let stream_holder = Arc::new(StreamHolder {
            stream: Mutex::new(Some(stream)),
        });

        Ok(Self {
            state,
            is_playing,
            chunks_completed,
            current_amplitude,
            sample_rate: target_rate,
            device_rate,
            _stream_holder: stream_holder,
        })
    }

    /// Get current playback amplitude (0.0 to 1.0)
    pub fn current_amplitude(&self) -> f32 {
        f32::from_bits(self.current_amplitude.load(Ordering::Relaxed))
    }

    /// Start the audio stream
    pub fn start(&self) -> Result<()> {
        if let Some(stream) = self._stream_holder.stream.lock().unwrap().as_ref() {
            stream.play()?;
        }
        info!("Audio playback started");
        Ok(())
    }

    /// Enqueue audio for playback
    /// Audio should be f32 samples normalized to [-1, 1]
    pub fn enqueue(&self, audio: Vec<f32>) {
        let mut state = self.state.lock().unwrap();

        // Resample if device rate differs from source rate
        let resampled = if self.device_rate != self.sample_rate {
            Self::resample_linear(&audio, self.sample_rate, self.device_rate)
        } else {
            audio
        };

        state.queue.push_back(resampled);
        self.is_playing.store(true, Ordering::SeqCst);
        debug!("Enqueued audio chunk, queue size: {}", state.queue.len() + 1);
    }

    /// Simple linear interpolation resampling
    fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate || input.is_empty() {
            return input.to_vec();
        }

        let ratio = from_rate as f64 / to_rate as f64;
        let output_len = (input.len() as f64 / ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(input.len() - 1);
            let frac = src_idx - idx0 as f64;

            let sample = input[idx0] as f64 * (1.0 - frac) + input[idx1] as f64 * frac;
            output.push(sample as f32);
        }

        output
    }

    /// Enqueue PCM16 audio (will be converted to f32)
    pub fn enqueue_pcm16(&self, pcm16: &[i16]) {
        let f32_audio: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();
        self.enqueue(f32_audio);
    }

    /// Enqueue WAV bytes (parses header and converts to f32)
    pub fn enqueue_wav(&self, wav_bytes: &[u8]) -> Result<()> {
        // Simple WAV parser - assumes 16-bit PCM
        if wav_bytes.len() < 44 {
            return Err(anyhow!("WAV data too short"));
        }

        // Read sample rate from WAV header (bytes 24-27)
        let wav_sample_rate = u32::from_le_bytes([
            wav_bytes[24],
            wav_bytes[25],
            wav_bytes[26],
            wav_bytes[27],
        ]);

        // Find data chunk
        let data_start = if &wav_bytes[36..40] == b"data" {
            44
        } else {
            // Search for data chunk
            let mut pos = 36;
            loop {
                if pos + 8 > wav_bytes.len() {
                    return Err(anyhow!("Could not find WAV data chunk"));
                }
                if &wav_bytes[pos..pos + 4] == b"data" {
                    let size = u32::from_le_bytes([
                        wav_bytes[pos + 4],
                        wav_bytes[pos + 5],
                        wav_bytes[pos + 6],
                        wav_bytes[pos + 7],
                    ]) as usize;
                    return self.enqueue_pcm16_bytes_with_rate(&wav_bytes[pos + 8..pos + 8 + size], wav_sample_rate);
                }
                let chunk_size = u32::from_le_bytes([
                    wav_bytes[pos + 4],
                    wav_bytes[pos + 5],
                    wav_bytes[pos + 6],
                    wav_bytes[pos + 7],
                ]) as usize;
                pos += 8 + chunk_size;
            }
        };

        self.enqueue_pcm16_bytes_with_rate(&wav_bytes[data_start..], wav_sample_rate)
    }

    fn enqueue_pcm16_bytes(&self, bytes: &[u8]) -> Result<()> {
        self.enqueue_pcm16_bytes_with_rate(bytes, self.sample_rate)
    }

    fn enqueue_pcm16_bytes_with_rate(&self, bytes: &[u8], source_rate: u32) -> Result<()> {
        if bytes.len() % 2 != 0 {
            return Err(anyhow!("PCM16 data length must be even"));
        }

        let pcm16: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let f32_audio: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();
        self.enqueue_with_rate(f32_audio, source_rate);
        Ok(())
    }

    /// Enqueue raw f32 PCM samples with specified source sample rate.
    /// Used for streaming TTS chunks.
    pub fn enqueue_pcm_f32(&self, samples: Vec<f32>, source_rate: u32) {
        self.enqueue_with_rate(samples, source_rate);
    }

    /// Enqueue audio with a specific source sample rate
    fn enqueue_with_rate(&self, audio: Vec<f32>, source_rate: u32) {
        let mut state = self.state.lock().unwrap();

        // Resample from source rate to device rate
        let resampled = if self.device_rate != source_rate {
            Self::resample_linear(&audio, source_rate, self.device_rate)
        } else {
            audio
        };

        state.queue.push_back(resampled);
        self.is_playing.store(true, Ordering::SeqCst);
        debug!("Enqueued audio chunk ({}Hz -> {}Hz), queue size: {}",
            source_rate, self.device_rate, state.queue.len() + 1);
    }

    /// Trigger fade-out and stop playback
    pub fn fade_out_and_stop(&self) {
        if self.is_playing.load(Ordering::SeqCst) {
            let mut state = self.state.lock().unwrap();
            let fade_samples = (self.device_rate * FADE_OUT_MS / 1000) as usize;
            state.fade_out = true;
            state.fade_samples_remaining = fade_samples;
            debug!("Initiated fade-out ({} samples)", fade_samples);
        }
    }

    /// Immediately stop and clear all audio
    pub fn stop_immediate(&self) {
        let mut state = self.state.lock().unwrap();
        state.queue.clear();
        state.current_chunk.clear();
        state.position = 0;
        self.is_playing.store(false, Ordering::SeqCst);
        state.fade_out = false;
        state.fade_samples_remaining = 0;
        debug!("Audio playback stopped immediately");
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::SeqCst)
    }

    /// Get number of fully completed chunks
    pub fn chunks_completed(&self) -> usize {
        self.chunks_completed.load(Ordering::SeqCst)
    }

    /// Reset completed chunks counter
    pub fn reset_chunks_completed(&self) {
        self.chunks_completed.store(0, Ordering::SeqCst);
    }

    /// Get queue length
    pub fn queue_len(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.queue.len() + if !state.current_chunk.is_empty() { 1 } else { 0 }
    }
}
