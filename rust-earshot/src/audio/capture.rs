//! Microphone audio capture using cpal

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::AudioConfig;

/// Audio capture from microphone
pub struct AudioCapture {
    stream: cpal::Stream,
    receiver: mpsc::Receiver<Vec<f32>>,
    frame_samples: usize,
    buffer: Vec<f32>,
}

impl AudioCapture {
    /// Create a new audio capture instance
    pub fn new(cfg: &AudioConfig) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No input device available"))?;

        info!("Input device: {}", device.name().unwrap_or_default());

        let config = cpal::StreamConfig {
            channels: cfg.channels,
            sample_rate: cpal::SampleRate(cfg.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let frame_samples = (cfg.sample_rate * cfg.frame_ms / 1000) as usize;
        debug!("Frame size: {} samples ({} ms)", frame_samples, cfg.frame_ms);

        // Channel for audio data
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(64);

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Send audio data to receiver
                let _ = tx.try_send(data.to_vec());
            },
            |err| warn!("Audio capture error: {}", err),
            None,
        )?;

        Ok(Self {
            stream,
            receiver: rx,
            frame_samples,
            buffer: Vec::with_capacity(frame_samples * 2),
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
            self.buffer.extend_from_slice(&data);
        }

        // Return frame if we have enough samples
        if self.buffer.len() >= self.frame_samples {
            let frame: Vec<f32> = self.buffer.drain(..self.frame_samples).collect();
            Some(frame)
        } else {
            None
        }
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
