"""
Silero VAD: ONNX runtime session for speech detection.
Assumes 16kHz PCM16, 20ms frames. Apply EMA smoothing in caller if desired.
"""
import onnxruntime as ort
import numpy as np


class TenVAD:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        # Create ONNX runtime session with CPU execution provider
        providers = ['CPUExecutionProvider']
        # Try CoreML on Apple Silicon if available
        if 'CoreMLExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CoreMLExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.sample_rate = sample_rate

        # Silero VAD state: (2, 1, 128) combined state
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr_tensor = np.array(sample_rate, dtype=np.int64)

    def reset(self):
        """Reset internal states (if model uses context)."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def prob_speech(self, pcm16: np.ndarray) -> float:
        """
        :param pcm16: int16 mono frame of length frame_samples
        :return: float probability 0..1
        """
        # Normalize int16 to float32 [-1, 1]
        audio_float = pcm16.astype(np.float32) / 32768.0

        # Silero VAD expects shape (1, samples)
        audio_input = audio_float.reshape(1, -1)

        # Run inference with state
        outputs = self.session.run(
            None,  # Get all outputs
            {
                'input': audio_input,
                'state': self._state,
                'sr': self._sr_tensor
            }
        )

        # outputs: [output, stateN]
        prob = float(outputs[0][0, 0])  # Shape is (1, 1)
        self._state = outputs[1]

        return np.clip(prob, 0.0, 1.0)
