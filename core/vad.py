"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


class SileroVAD:
    """Silero VAD wrapper — detects speech in audio chunks."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.sample_rate = sample_rate
        self._model = None

    def load(self) -> None:
        """Load Silero VAD model."""
        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._model.eval()
        log.info("Silero VAD loaded")

    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio chunk contains speech using energy + Silero VAD."""
        if self._model is None:
            self.load()
        audio = np.array(audio, dtype=np.float32, copy=True)

        # Quick energy check — skip dead silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.001:
            return False

        tensor = torch.from_numpy(audio)
        if tensor.dim() > 1:
            tensor = tensor.mean(dim=-1)
        # Resample to 16kHz if needed
        if self.sample_rate != 16000:
            import torchaudio
            tensor = tensor.unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, self.sample_rate, 16000)
            tensor = tensor.squeeze(0)
        # Silero expects 512-sample chunks at 16kHz
        # Feed sequentially, return True if any chunk has speech
        win = 512
        if len(tensor) < win:
            tensor = torch.nn.functional.pad(tensor, (0, win - len(tensor)))
        detected = False
        for start in range(0, len(tensor) - win + 1, win):
            chunk = tensor[start:start + win]
            prob = self._model(chunk, 16000).item()
            if prob >= self.threshold:
                detected = True
                break
        return detected

    def reset(self) -> None:
        """Reset VAD internal state between segments."""
        if self._model is not None:
            self._model.reset_states()
