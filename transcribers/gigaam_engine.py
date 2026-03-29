"""GigaAM transcriber — Sber's Russian ASR model."""

from __future__ import annotations

import logging

import numpy as np

from transcribers.base import AbstractTranscriber, TranscriptionResult

log = logging.getLogger(__name__)


class GigaAMTranscriber(AbstractTranscriber):
    """GigaAM v3 transcription engine (CTC or RNNT)."""

    def __init__(
        self,
        model_type: str = "ctc",  # ctc | rnnt
        device: str = "auto",
    ):
        self.model_type = model_type
        self.device = device
        self._model = None

    def load(self) -> None:
        import gigaam

        log.info("Loading GigaAM %s...", self.model_type)
        self._model = gigaam.load_model(self.model_type)
        if self.device == "cuda":
            self._model = self._model.cuda()
        log.info("GigaAM model loaded (%s)", self.model_type)

    def transcribe(self, audio: np.ndarray, sr: int) -> list[TranscriptionResult]:
        if self._model is None:
            self.load()

        import torch

        # GigaAM expects 16kHz mono float32 tensor
        tensor = torch.from_numpy(audio).float().unsqueeze(0)
        if self.device == "cuda":
            tensor = tensor.cuda()

        with torch.no_grad():
            recognition = self._model.transcribe(tensor)

        results = []
        for item in recognition:
            text = item.text.strip() if hasattr(item, "text") else str(item).strip()
            if text:
                start = getattr(item, "start", 0.0)
                end = getattr(item, "end", len(audio) / sr)
                results.append(TranscriptionResult(
                    text=text,
                    start=start,
                    end=end,
                    language="ru",
                ))
        return results

    def unload(self) -> None:
        self._model = None
        log.info("GigaAM model unloaded")
