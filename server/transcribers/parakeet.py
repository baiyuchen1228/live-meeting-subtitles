"""
Parakeet TDT v3 transcriber using NVIDIA NeMo.
Optimized for high-speed transcription on NVIDIA GPUs.

Note: RTX 50 series (Blackwell) requires PyTorch nightly with CUDA 12.8+.
"""

import time
import numpy as np

from .base import BaseTranscriber, TranscriptionResult

try:
    import torch
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False


class ParakeetTranscriber(BaseTranscriber):
    """
    Parakeet TDT v3 transcription using NVIDIA NeMo.

    Parakeet is ~2x faster than Whisper with similar quality.
    Supports 25 European languages including English and Russian.
    Built-in punctuation and capitalization.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
        language: str | None = None,  # Auto-detect by default
    ):
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NeMo toolkit required: pip install nemo_toolkit[asr]\n"
                "For RTX 5090: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
            )

        self.model_name = model_name
        self.device = device
        self.language = language
        self._model = None

    def load(self) -> None:
        """Load the model (call once at startup)."""
        if self._model is not None:
            return

        print(f"Loading Parakeet model: {self.model_name} on {self.device}...")
        start = time.time()

        self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        # Disable CUDA graphs (workaround for PyTorch nightly compatibility)
        self._disable_cuda_graphs()

        # We call the model directly (see _transcribe_direct), so set up what
        # NeMo's transcribe() would: eval mode, no dither noise, no padding
        self._model.eval()
        self._model.preprocessor.featurizer.dither = 0.0
        self._model.preprocessor.featurizer.pad_to = 0

        # Loading peaks at ~2x the model size; release that cache for the translator
        torch.cuda.empty_cache()

        print(f"Parakeet model loaded in {time.time() - start:.1f}s")

    def _disable_cuda_graphs(self) -> None:
        """Disable CUDA graphs for PyTorch nightly compatibility."""
        if self._model is None:
            return

        try:
            decoding_cfg = self._model.cfg.decoding.copy()
            with open_dict(decoding_cfg):
                decoding_cfg.greedy.use_cuda_graph_decoder = False
                decoding_cfg.greedy.loop_labels = False
            self._model.change_decoding_strategy(decoding_cfg)
        except Exception as e:
            print(f"Warning: Could not disable CUDA graphs: {e}")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult | None:
        """
        Transcribe audio chunk.

        Args:
            audio: Float32 numpy array, mono, normalized to [-1, 1]
            sample_rate: Sample rate (should be 16000 for Parakeet)

        Returns:
            TranscriptionResult or None if no speech detected
        """
        if self._model is None:
            self.load()

        # Skip if too quiet
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.005:
            return None

        start = time.time()

        try:
            text = self._transcribe_direct(audio)
        except Exception as e:
            # Fall back to NeMo's high-level API (slower: builds a dataloader per call)
            print(f"Parakeet direct inference failed, using transcribe(): {e}")
            try:
                results = self._model.transcribe([audio], verbose=False)
                text = results[0].text if results else ""
            except Exception as e:
                print(f"Parakeet transcription error: {e}")
                return None

        text = text.strip()
        if not text:
            return None

        return TranscriptionResult(
            text=text,
            language=self.language or "unknown",
            confidence=1.0,  # Parakeet doesn't provide confidence
            processing_time=time.time() - start,
        )

    def _transcribe_direct(self, audio: np.ndarray) -> str:
        """Run preprocessor -> encoder -> decoder on an in-memory array.
        2-3x faster than transcribe(), which writes temp files and builds a dataloader per call."""
        model = self._model
        with torch.inference_mode():
            signal = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).to(model.device)
            length = torch.tensor([len(audio)], device=model.device)
            encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=length)
            hyps = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len
            )
        if isinstance(hyps, tuple):  # older NeMo returns (best, all)
            hyps = hyps[0]
        return hyps[0].text if hyps else ""

    def unload(self) -> None:
        """Unload model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    def is_loaded(self) -> bool:
        return self._model is not None
