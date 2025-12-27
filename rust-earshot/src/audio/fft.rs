//! Real-time FFT processing for audio visualization

use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

/// Number of display bins for the radial visualizer
pub const NUM_DISPLAY_BINS: usize = 64;

/// FFT snapshot containing frequency data for visualization
#[derive(Clone, Debug)]
pub struct FFTSnapshot {
    /// Frequency bin magnitudes (0.0-1.0 normalized)
    pub magnitudes: [f32; NUM_DISPLAY_BINS],
    /// Index of the dominant (peak) frequency bin
    pub dominant_bin: u8,
    /// Low frequency energy (0-500Hz) for orb pulsing
    pub bass_energy: f32,
    /// Mid frequency energy (500-2000Hz) for ring rotation
    pub mid_energy: f32,
    /// High frequency energy (2000Hz+) for particle effects
    pub high_energy: f32,
}

impl Default for FFTSnapshot {
    fn default() -> Self {
        Self {
            magnitudes: [0.0; NUM_DISPLAY_BINS],
            dominant_bin: 0,
            bass_energy: 0.0,
            mid_energy: 0.0,
            high_energy: 0.0,
        }
    }
}

/// Real-time FFT processor for audio visualization
pub struct FFTProcessor {
    fft: Arc<dyn rustfft::Fft<f32>>,
    fft_size: usize,
    sample_rate: u32,
    window: Vec<f32>,
    input_buffer: Vec<Complex<f32>>,
    output_buffer: Vec<Complex<f32>>,
    magnitude_buffer: Vec<f32>,
    smoothed_magnitudes: [f32; NUM_DISPLAY_BINS],
    /// Decay factor for exponential smoothing (0.0-1.0)
    smoothing_decay: f32,
    /// Frame counter for periodic logging
    frame_count: u32,
    /// Running average of max magnitude for adaptive scaling
    avg_max_magnitude: f32,
}

impl FFTProcessor {
    /// Create a new FFT processor
    ///
    /// # Arguments
    /// * `fft_size` - FFT size (should be power of 2, e.g., 128, 256)
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(fft_size: usize, sample_rate: u32) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        // Hann window for smooth spectral leakage
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos())
            })
            .collect();

        Self {
            fft,
            fft_size,
            sample_rate,
            window,
            input_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            output_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            magnitude_buffer: vec![0.0; fft_size / 2],
            smoothed_magnitudes: [0.0; NUM_DISPLAY_BINS],
            smoothing_decay: 0.85, // Slower decay for smoother visualization
            frame_count: 0,
            avg_max_magnitude: 0.1, // Start with a reasonable baseline
        }
    }

    /// Process audio samples and return FFT snapshot
    ///
    /// # Arguments
    /// * `samples` - Audio samples (mono, f32). Will use up to fft_size samples.
    pub fn process(&mut self, samples: &[f32]) -> FFTSnapshot {
        self.frame_count += 1;

        // Handle case where we have fewer samples than FFT size
        let num_samples = samples.len().min(self.fft_size);

        // Apply window and convert to complex
        for i in 0..self.fft_size {
            let sample = if i < num_samples { samples[i] } else { 0.0 };
            self.input_buffer[i] = Complex::new(sample * self.window[i], 0.0);
        }

        // Perform FFT
        self.output_buffer.copy_from_slice(&self.input_buffer);
        self.fft.process(&mut self.output_buffer);

        // Calculate magnitudes (only first half - positive frequencies)
        let half_size = self.fft_size / 2;
        for (i, c) in self.output_buffer[..half_size].iter().enumerate() {
            self.magnitude_buffer[i] = c.norm();
        }

        // Logarithmic binning to display bins - use RMS average instead of max
        let mut magnitudes = [0.0f32; NUM_DISPLAY_BINS];
        for i in 0..NUM_DISPLAY_BINS {
            let (start, end) = self.log_bin_range(i);
            if start < end && end <= half_size {
                // Use RMS (root mean square) for smoother response
                let sum_sq: f32 = self.magnitude_buffer[start..end]
                    .iter()
                    .map(|x| x * x)
                    .sum();
                let count = (end - start) as f32;
                magnitudes[i] = (sum_sq / count).sqrt();
            }
        }

        // Calculate current max magnitude
        let current_max = magnitudes.iter().copied().fold(0.0001f32, f32::max);

        // Update running average of max (slow adaptation)
        self.avg_max_magnitude = self.avg_max_magnitude * 0.99 + current_max * 0.01;

        // Use the larger of current max or running average for normalization
        // This prevents background noise from causing big spikes
        let norm_factor = self.avg_max_magnitude.max(current_max * 0.5).max(0.001);

        // Apply noise floor - anything below 5% of norm_factor is zeroed
        let noise_floor = norm_factor * 0.05;

        // Normalize magnitudes with noise floor and boost for visibility
        let boost = 4.0; // Boost factor to make visualization more dynamic
        for m in &mut magnitudes {
            *m = (((*m - noise_floor).max(0.0) / norm_factor) * boost).min(1.0);
        }

        // Apply exponential smoothing with slower decay
        for (i, &m) in magnitudes.iter().enumerate() {
            // Moderate attack, slow decay
            let attack_factor = 0.4; // How fast to respond to increases
            if m > self.smoothed_magnitudes[i] {
                self.smoothed_magnitudes[i] =
                    self.smoothed_magnitudes[i] * (1.0 - attack_factor) + m * attack_factor;
            } else {
                self.smoothed_magnitudes[i] =
                    self.smoothed_magnitudes[i] * self.smoothing_decay + m * (1.0 - self.smoothing_decay);
            }
            // Hard floor - anything below 0.02 is zeroed to prevent tiny residual spikes
            if self.smoothed_magnitudes[i] < 0.02 {
                self.smoothed_magnitudes[i] = 0.0;
            }
        }

        // Periodic logging (every ~60 frames, roughly once per second at 60fps)
        if self.frame_count % 60 == 0 {
            let bass_raw: f32 = magnitudes[0..8].iter().sum::<f32>() / 8.0;
            let mid_raw: f32 = magnitudes[8..32].iter().sum::<f32>() / 24.0;
            let high_raw: f32 = magnitudes[32..64].iter().sum::<f32>() / 32.0;
            tracing::debug!(
                "[FFT] max={:.4} avg_max={:.4} norm={:.4} | raw bass={:.3} mid={:.3} high={:.3} | smoothed bass={:.3} mid={:.3} high={:.3}",
                current_max, self.avg_max_magnitude, norm_factor,
                bass_raw, mid_raw, high_raw,
                self.smoothed_magnitudes[0..8].iter().sum::<f32>() / 8.0,
                self.smoothed_magnitudes[8..32].iter().sum::<f32>() / 24.0,
                self.smoothed_magnitudes[32..64].iter().sum::<f32>() / 32.0,
            );
        }

        // Calculate frequency band energies
        // Bass: bins 0-7 (~0-500Hz at 16kHz, 128-point FFT)
        let bass_energy = self.smoothed_magnitudes[0..8].iter().sum::<f32>() / 8.0;
        // Mids: bins 8-31 (~500-2000Hz)
        let mid_energy = self.smoothed_magnitudes[8..32].iter().sum::<f32>() / 24.0;
        // Highs: bins 32-63 (~2000Hz+)
        let high_energy = self.smoothed_magnitudes[32..64].iter().sum::<f32>() / 32.0;

        // Find dominant frequency bin
        let dominant_bin = self.smoothed_magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        FFTSnapshot {
            magnitudes: self.smoothed_magnitudes,
            dominant_bin,
            bass_energy,
            mid_energy,
            high_energy,
        }
    }

    /// Calculate bin range for a display bin - voice-only distribution
    /// Only uses the first 45% of FFT bins (where voice energy is) spread across all display bins
    fn log_bin_range(&self, display_bin: usize) -> (usize, usize) {
        let half_size = self.fft_size / 2;

        // Only use first 45% of FFT spectrum (voice range), spread across all 64 display bins
        let voice_range = 0.45;
        let max_fft_bin = (half_size as f32 * voice_range) as usize;

        let t = display_bin as f32 / NUM_DISPLAY_BINS as f32;
        let t_end = (display_bin + 1) as f32 / NUM_DISPLAY_BINS as f32;

        // Linear mapping across the voice range
        let start = (t * max_fft_bin as f32) as usize;
        let end = (t_end * max_fft_bin as f32) as usize;

        (start.max(1), end.max(start + 1).min(half_size))
    }

    /// Get the frequency in Hz for a given display bin
    #[allow(dead_code)]
    pub fn bin_to_frequency(&self, bin: usize) -> f32 {
        let (start, end) = self.log_bin_range(bin);
        let center = (start + end) / 2;
        center as f32 * self.sample_rate as f32 / self.fft_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_processor_basic() {
        let mut processor = FFTProcessor::new(128, 16000);

        // Generate a simple sine wave at 1000Hz
        let samples: Vec<f32> = (0..128)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin())
            .collect();

        let snapshot = processor.process(&samples);

        // Should have some energy
        assert!(snapshot.magnitudes.iter().any(|&m| m > 0.0));
        // Mid frequencies should have most energy for 1000Hz tone
        assert!(snapshot.mid_energy > snapshot.bass_energy);
    }

    #[test]
    fn test_empty_input() {
        let mut processor = FFTProcessor::new(128, 16000);
        let snapshot = processor.process(&[]);

        // Should not panic, magnitudes should be low/zero
        assert!(snapshot.magnitudes.iter().all(|&m| m <= 1.0));
    }
}
