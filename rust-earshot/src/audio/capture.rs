//! Microphone audio capture using cpal

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::AudioConfig;

fn device_name_matches(device: &cpal::Device, needle: &str) -> bool {
    let name = match device.name() {
        Ok(n) => n,
        Err(_) => return false,
    };
    name.to_lowercase().contains(&needle.to_lowercase())
}

/// List available input devices
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    match host.input_devices() {
        Ok(input_devices) => {
            for device in input_devices {
                let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
                info!("Input device: {}", name);
                devices.push(name);
            }
        }
        Err(e) => {
            warn!("Failed to list input devices: {}", e);
        }
    }
    devices
}

/// Audio capture from microphone with optional resampling
pub struct AudioCapture {
    stream: cpal::Stream,
    receiver: mpsc::Receiver<Vec<f32>>,
    frame_samples: usize,
    buffer: Vec<f32>,
    resample: Option<(u32, u32)>,
}

impl AudioCapture {
    /// Create a new audio capture instance
    pub fn new(cfg: &AudioConfig) -> Result<Self> {
        let host = cpal::default_host();
        let device = if let Some(preferred) = cfg.input_device.as_deref() {
            let devices: Vec<cpal::Device> = host.input_devices()?.collect();
            if let Some(found) = devices.iter().find(|d| device_name_matches(d, preferred)) {
                info!("Using input device: {}", found.name().unwrap_or_default());
                found.clone()
            } else {
                let names = devices
                    .iter()
                    .map(|d| d.name().unwrap_or_else(|_| "<unknown>".to_string()))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(anyhow!(
                    "Input device '{}' not found. Available: {}",
                    preferred,
                    names
                ));
            }
        } else {
            host.default_input_device()
                .ok_or_else(|| anyhow!("No input device available"))?
        };

        info!("Input device: {}", device.name().unwrap_or_default());

        let target_rate = cfg.sample_rate;

        let mut config = cpal::StreamConfig {
            channels: cfg.channels,
            sample_rate: cpal::SampleRate(target_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Use device's default config and resample as needed
        let resample = match device.default_input_config() {
            Ok(default_config) => {
                let device_rate = default_config.sample_rate().0;
                let device_channels = default_config.channels();

                config.sample_rate = cpal::SampleRate(device_rate);
                config.channels = device_channels;

                if device_rate == target_rate {
                    info!("Device supports {}Hz natively ({} channels)", target_rate, device_channels);
                    None
                } else {
                    info!(
                        "Device uses {}Hz ({} channels), will resample to {}Hz",
                        device_rate, device_channels, target_rate
                    );
                    Some((device_rate, target_rate))
                }
            }
            Err(e) => {
                warn!("Could not get default config: {}, trying requested config", e);
                None
            }
        };

        let frame_samples = (target_rate * cfg.frame_ms / 1000) as usize;
        debug!(
            "Frame size: {} samples ({} ms at {}Hz)",
            frame_samples, cfg.frame_ms, target_rate
        );

        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(64);
        let channels = config.channels as usize;

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
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
        })
    }

    /// Create audio capture for a specific device by name
    pub fn with_device(device_name: &str, cfg: &AudioConfig) -> Result<Self> {
        let mut cfg = cfg.clone();
        cfg.input_device = Some(device_name.to_string());
        Self::new(&cfg)
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
    pub fn next_frame(&mut self) -> Option<Vec<f32>> {
        loop {
            while self.buffer.len() < self.frame_samples {
                match self.receiver.recv() {
                    Ok(data) => {
                        if let Some((from_rate, to_rate)) = self.resample {
                            let resampled = Self::resample_linear(&data, from_rate, to_rate);
                            self.buffer.extend_from_slice(&resampled);
                        } else {
                            self.buffer.extend_from_slice(&data);
                        }
                    }
                    Err(_) => return None,
                }
            }

            if self.buffer.len() >= self.frame_samples {
                let frame: Vec<f32> = self.buffer.drain(..self.frame_samples).collect();
                return Some(frame);
            }
        }
    }

    /// Try to get the next frame (non-blocking)
    pub fn try_next_frame(&mut self) -> Option<Vec<f32>> {
        while let Ok(data) = self.receiver.try_recv() {
            if let Some((from_rate, to_rate)) = self.resample {
                let resampled = Self::resample_linear(&data, from_rate, to_rate);
                self.buffer.extend_from_slice(&resampled);
            } else {
                self.buffer.extend_from_slice(&data);
            }
        }

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
