//! Microphone audio capture using cpal

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::AudioConfig;

/// Audio capture from microphone with optional resampling
pub struct AudioCapture {
    stream: cpal::Stream,
    receiver: mpsc::Receiver<Vec<f32>>,
    frame_samples: usize,
    buffer: Vec<f32>,
    /// Resampling state: (device_rate, target_rate)
    resample: Option<(u32, u32)>,
    resample_buffer: Vec<f32>,
}

impl AudioCapture {
    /// Create a new audio capture instance
    pub fn new(cfg: &AudioConfig) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No input device available"))?;

        info!("Input device: {}", device.name().unwrap_or_default());

        let target_rate = cfg.sample_rate;

        // Try to use the requested config first
        let mut config = cpal::StreamConfig {
            channels: cfg.channels,
            sample_rate: cpal::SampleRate(target_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Check if device supports our config, if not find a compatible one
        let resample = match device.supported_input_configs() {
            Ok(supported) => {
                let supported: Vec<_> = supported.collect();
                let has_exact = supported.iter().any(|c| {
                    c.channels() == cfg.channels &&
                    c.min_sample_rate().0 <= target_rate &&
                    c.max_sample_rate().0 >= target_rate
                });

                if has_exact {
                    info!("Device supports {}Hz natively", target_rate);
                    None
                } else {
                    // Find a supported rate we can resample from (prefer 48kHz, then 44.1kHz)
                    let preferred_rates = [48000u32, 44100, 96000, 22050];
                    let mut found_rate = None;

                    for rate in preferred_rates {
                        if supported.iter().any(|c| {
                            c.channels() >= cfg.channels &&
                            c.min_sample_rate().0 <= rate &&
                            c.max_sample_rate().0 >= rate
                        }) {
                            found_rate = Some(rate);
                            break;
                        }
                    }

                    if let Some(device_rate) = found_rate {
                        info!("Device uses {}Hz, will resample to {}Hz", device_rate, target_rate);
                        config.sample_rate = cpal::SampleRate(device_rate);
                        Some((device_rate, target_rate))
                    } else {
                        // Try default config
                        if let Ok(default) = device.default_input_config() {
                            let device_rate = default.sample_rate().0;
                            info!("Using device default {}Hz, will resample to {}Hz", device_rate, target_rate);
                            config.sample_rate = cpal::SampleRate(device_rate);
                            config.channels = default.channels().min(cfg.channels);
                            Some((device_rate, target_rate))
                        } else {
                            warn!("Could not determine supported config, trying requested config");
                            None
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Could not query supported configs: {}, trying requested", e);
                None
            }
        };

        let frame_samples = (target_rate * cfg.frame_ms / 1000) as usize;
        debug!("Frame size: {} samples ({} ms at {}Hz)", frame_samples, cfg.frame_ms, target_rate);

        // Channel for audio data
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(64);
        let channels = config.channels as usize;

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert to mono if needed
                let mono: Vec<f32> = if channels > 1 {
                    data.chunks(channels)
                        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                        .collect()
                } else {
                    data.to_vec()
                };
                let _ = tx.try_send(mono);
            },
            |err| warn!("Audio capture error: {}", err),
            None,
        )?;

        Ok(Self {
            stream,
            receiver: rx,
            frame_samples,
            buffer: Vec::with_capacity(frame_samples * 2),
            resample,
            resample_buffer: Vec::new(),
        })
    }

    /// Start audio capture
    pub fn start(&self) -> Result<()> {
        self.stream.play()?;
        info!("Audio capture started");
        Ok(())
    }

    /// Stop audio capture
    pub fn stop(&self) -> Result<()> {
        self.stream.pause()?;
        info!("Audio capture stopped");
        Ok(())
    }

    /// Get the next frame of audio (blocking)
    /// Returns f32 samples normalized to [-1, 1]
    pub fn next_frame(&mut self) -> Option<Vec<f32>> {
        loop {
            // Try to fill buffer from received data
            while self.buffer.len() < self.frame_samples {
                match self.receiver.recv() {
                    Ok(data) => self.buffer.extend_from_slice(&data),
                    Err(_) => return None,
                }
            }

            // Extract one frame
            if self.buffer.len() >= self.frame_samples {
                let frame: Vec<f32> = self.buffer.drain(..self.frame_samples).collect();
                return Some(frame);
            }
        }
    }

    /// Try to get the next frame (non-blocking)
    pub fn try_next_frame(&mut self) -> Option<Vec<f32>> {
        // Drain all available data
        while let Ok(data) = self.receiver.try_recv() {
            if let Some((from_rate, to_rate)) = self.resample {
                // Resample the incoming data
                let resampled = Self::resample_linear(&data, from_rate, to_rate);
                self.buffer.extend_from_slice(&resampled);
            } else {
                self.buffer.extend_from_slice(&data);
            }
        }

        // Return frame if we have enough samples
        if self.buffer.len() >= self.frame_samples {
            let frame: Vec<f32> = self.buffer.drain(..self.frame_samples).collect();
            Some(frame)
        } else {
            None
        }
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

    /// Convert f32 samples to i16 PCM
    pub fn to_pcm16(samples: &[f32]) -> Vec<i16> {
        samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }

    /// Convert f32 samples to PCM16 bytes (little-endian)
    pub fn to_pcm16_bytes(samples: &[f32]) -> Vec<u8> {
        let pcm16 = Self::to_pcm16(samples);
        pcm16.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }
}
