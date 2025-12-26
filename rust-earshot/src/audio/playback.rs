//! Audio playback with fade-out support for interruption

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

const FADE_OUT_MS: u32 = 250;

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
    sample_rate: u32,
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

        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let state = Arc::new(Mutex::new(PlaybackState {
            queue: VecDeque::new(),
            current_chunk: Vec::new(),
            position: 0,
            fade_out: false,
            fade_samples_remaining: 0,
        }));

        let is_playing = Arc::new(AtomicBool::new(false));
        let chunks_completed = Arc::new(AtomicUsize::new(0));

        let state_clone = state.clone();
        let is_playing_clone = is_playing.clone();
        let chunks_completed_clone = chunks_completed.clone();
        let fade_samples = (sample_rate * FADE_OUT_MS / 1000) as usize;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut state = state_clone.lock().unwrap();

                for sample in data.iter_mut() {
                    if state.fade_out && state.fade_samples_remaining > 0 {
                        // Fade out current audio
                        let fade_factor = state.fade_samples_remaining as f32 / fade_samples as f32;
                        if state.position < state.current_chunk.len() {
                            *sample = state.current_chunk[state.position] * fade_factor;
                            state.position += 1;
                        } else {
                            *sample = 0.0;
                        }
                        state.fade_samples_remaining -= 1;

                        if state.fade_samples_remaining == 0 {
                            // Fade complete, clear everything
                            state.queue.clear();
                            state.current_chunk.clear();
                            state.position = 0;
                            is_playing_clone.store(false, Ordering::SeqCst);
                            state.fade_out = false;
                        }
                    } else if state.position < state.current_chunk.len() {
                        // Play current chunk
                        *sample = state.current_chunk[state.position];
                        state.position += 1;
                    } else if let Some(next_chunk) = state.queue.pop_front() {
                        // Move to next chunk
                        chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                        state.current_chunk = next_chunk;
                        state.position = 0;
                        if !state.current_chunk.is_empty() {
                            *sample = state.current_chunk[state.position];
                            state.position += 1;
                        } else {
                            *sample = 0.0;
                        }
                    } else {
                        // Nothing to play - we've finished the current chunk and queue is empty
                        *sample = 0.0;
                        if is_playing_clone.load(Ordering::SeqCst) {
                            // Clear the finished chunk and mark as not playing
                            state.current_chunk.clear();
                            state.position = 0;
                            is_playing_clone.store(false, Ordering::SeqCst);
                        }
                    }
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
            sample_rate,
            _stream_holder: stream_holder,
        })
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
        state.queue.push_back(audio);
        self.is_playing.store(true, Ordering::SeqCst);
        debug!("Enqueued audio chunk, queue size: {}", state.queue.len() + 1);
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
                    return self.enqueue_pcm16_bytes(&wav_bytes[pos + 8..pos + 8 + size]);
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

        self.enqueue_pcm16_bytes(&wav_bytes[data_start..])
    }

    fn enqueue_pcm16_bytes(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() % 2 != 0 {
            return Err(anyhow!("PCM16 data length must be even"));
        }

        let pcm16: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        self.enqueue_pcm16(&pcm16);
        Ok(())
    }

    /// Trigger fade-out and stop playback
    pub fn fade_out_and_stop(&self) {
        if self.is_playing.load(Ordering::SeqCst) {
            let mut state = self.state.lock().unwrap();
            let fade_samples = (self.sample_rate * FADE_OUT_MS / 1000) as usize;
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
