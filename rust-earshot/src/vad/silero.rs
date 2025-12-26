//! Silero VAD - ONNX-based Voice Activity Detection

use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};
use ort::{session::Session, value::Tensor};
use tracing::debug;

const WINDOW_SIZE: usize = 512; // 32ms at 16kHz
const CONTEXT_SIZE: usize = 64;

/// Silero Voice Activity Detection model
pub struct SileroVad {
    session: Session,
    state: ArrayD<f32>,
    context: Vec<f32>,
    input_buffer: Vec<f32>,
    sample_rate: u32,
}

impl SileroVad {
    /// Create a new Silero VAD instance
    pub fn new(model_path: &str, sample_rate: u32) -> Result<Self> {
        debug!("Loading Silero VAD model from {}", model_path);

        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .with_intra_threads(1)
            .context("Failed to set intra threads")?
            .with_inter_threads(1)
            .context("Failed to set inter threads")?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        // VAD state [2, 1, 128]
        let state = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 128]));
        let context = vec![0.0f32; CONTEXT_SIZE];
        let input_buffer = vec![0.0f32; CONTEXT_SIZE + WINDOW_SIZE];

        debug!("Silero VAD model loaded successfully");

        Ok(Self {
            session,
            state,
            context,
            input_buffer,
            sample_rate,
        })
    }

    /// Process audio samples and return speech probability
    /// Input: f32 samples normalized to [-1, 1]
    /// Output: speech probability [0, 1]
    pub fn process(&mut self, samples: &[f32]) -> Result<f32> {
        // Silero expects 512 samples (32ms at 16kHz)
        if samples.len() != WINDOW_SIZE {
            // If we get a different frame size, resample or pad/truncate
            let mut chunk = vec![0.0f32; WINDOW_SIZE];
            let copy_len = samples.len().min(WINDOW_SIZE);
            chunk[..copy_len].copy_from_slice(&samples[..copy_len]);
            return self.process_chunk(&chunk);
        }

        self.process_chunk(samples)
    }

    fn process_chunk(&mut self, chunk: &[f32]) -> Result<f32> {
        // Build input: context + chunk
        self.input_buffer[..CONTEXT_SIZE].copy_from_slice(&self.context);
        self.input_buffer[CONTEXT_SIZE..].copy_from_slice(chunk);

        // Update context for next iteration
        self.context.copy_from_slice(&self.input_buffer[WINDOW_SIZE..]);

        // Create tensors
        let input_tensor = Tensor::from_array((
            [1, CONTEXT_SIZE + WINDOW_SIZE],
            self.input_buffer.clone(),
        ))?;

        let state_tensor = Tensor::from_array((
            self.state.shape().to_vec(),
            self.state.as_slice().unwrap().to_vec(),
        ))?;

        let sr_tensor = Tensor::from_array(([1], vec![self.sample_rate as i64]))?;

        // Run inference
        let outputs = self.session.run(ort::inputs! {
            "input" => input_tensor,
            "state" => state_tensor,
            "sr" => sr_tensor
        })?;

        // Extract probability
        let (_, prob_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob = prob_data[0];

        // Update LSTM state
        let (shape, new_state_data) = outputs["stateN"].try_extract_tensor::<f32>()?;
        self.state = ArrayD::from_shape_vec(shape.to_ixdyn(), new_state_data.to_vec())?;

        Ok(prob)
    }

    /// Reset the VAD state
    pub fn reset(&mut self) {
        self.state = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 128]));
        self.context = vec![0.0f32; CONTEXT_SIZE];
        debug!("VAD state reset");
    }

    /// Get the expected window size in samples
    pub fn window_size(&self) -> usize {
        WINDOW_SIZE
    }
}
