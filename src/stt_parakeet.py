"""
NVIDIA Parakeet (NeMo) wrapper. Transcribes PCM16 bytes into text + word timestamps.
Significantly faster than Whisper with comparable accuracy for English.
"""
import os
import numpy as np
import torch
from typing import Dict, Any

# Disable CUDA graphs before importing NeMo
os.environ["NEMO_DISABLE_CUDA_GRAPH_DECODER"] = "1"


class ParakeetSTT:
    def __init__(self, cfg):
        self.model_name = cfg.get("model_name", "nvidia/parakeet-tdt-0.6b-v2")
        self.device = cfg.get("device", "cuda")

        # Import NeMo here to allow graceful fallback if not installed
        import nemo.collections.asr as nemo_asr

        # Load model
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )

        # Disable CUDA graphs at multiple levels
        self._disable_cuda_graphs()

        # Move to device if specified
        if self.device == "cuda":
            self.model = self.model.cuda()

        # Set to eval mode
        self.model.eval()

    def _disable_cuda_graphs(self):
        """Disable CUDA graphs at all levels to avoid compatibility issues (especially WSL2)."""
        from omegaconf import OmegaConf

        # Try to change decoding strategy to disable CUDA graphs
        try:
            # Get current decoding config and modify it
            if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'decoding'):
                decoding_cfg = OmegaConf.to_container(self.model.cfg.decoding, resolve=True)
                decoding_cfg['greedy'] = decoding_cfg.get('greedy', {})
                decoding_cfg['greedy']['loop_labels'] = False  # Disable loop labels which uses CUDA graphs
                decoding_cfg['greedy']['use_cuda_graph_decoder'] = False
                self.model.change_decoding_strategy(OmegaConf.create(decoding_cfg))
                print("Disabled CUDA graphs via decoding strategy")
        except Exception as e:
            print(f"Warning: Could not modify decoding config: {e}")

        # More aggressive: monkey-patch the decoding computer to not use CUDA graphs
        if hasattr(self.model, 'decoding'):
            decoding = self.model.decoding
            if hasattr(decoding, 'decoding'):
                inner = decoding.decoding
                if hasattr(inner, 'use_cuda_graph_decoder'):
                    inner.use_cuda_graph_decoder = False
                if hasattr(inner, 'loop_labels'):
                    inner.loop_labels = False

                # Patch the decoding computer's __call__ to skip CUDA graphs
                if hasattr(inner, 'decoding_computer') and inner.decoding_computer is not None:
                    computer = inner.decoding_computer
                    if hasattr(computer, 'use_cuda_graphs'):
                        computer.use_cuda_graphs = False
                    if hasattr(computer, '_use_cuda_graphs'):
                        computer._use_cuda_graphs = False

                    # Override the cuda_graphs_impl to use non_cuda_graphs_impl
                    if hasattr(computer, 'non_cuda_graphs_impl') and hasattr(computer, 'cuda_graphs_impl'):
                        computer.cuda_graphs_impl = computer.non_cuda_graphs_impl
                        print("Patched decoding_computer to use non-CUDA graphs implementation")

    def transcribe(self, pcm16_bytes: bytes, language: str = None, beam_size: int = 1, word_timestamps: bool = True) -> Dict[str, Any]:
        """
        Transcribe PCM16 audio bytes.

        Returns dict: {"text": str, "words": [{"w": str, "start_s": float, "end_s": float}, ...]}

        Note: Parakeet is English-only, so language parameter is ignored.
        """
        # Convert PCM16 bytes to float32 array normalized to [-1, 1]
        pcm16_array = np.frombuffer(pcm16_bytes, dtype=np.int16)
        audio_float = pcm16_array.astype(np.float32) / 32768.0

        # Transcribe with timestamps - pass numpy array directly
        with torch.no_grad():
            output = self.model.transcribe(
                audio_float,
                timestamps=word_timestamps,
                verbose=False
            )

        # Handle different return types
        if isinstance(output, tuple):
            output = output[0]

        # Extract results - output might be a list or single item
        if isinstance(output, list):
            result = output[0] if output else ""
        else:
            result = output

        # Get text
        if hasattr(result, 'text'):
            text = result.text
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)

        # Extract word timestamps if available
        all_words = []
        if word_timestamps and hasattr(result, 'timestamp') and result.timestamp:
            word_stamps = result.timestamp.get('word', [])
            for w in word_stamps:
                all_words.append({
                    "w": w.get('word', w.get('segment', '')).strip(),
                    "start_s": w.get('start', 0.0),
                    "end_s": w.get('end', 0.0)
                })

        return {
            "text": text.strip() if isinstance(text, str) else str(text),
            "words": all_words
        }
