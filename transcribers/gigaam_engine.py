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

        import tempfile
        import soundfile as sf

        # GigaAM expects a file path — save to temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            sf.write(tmp_path, audio, sr)

        try:
            recognition = self._model.transcribe(tmp_path)
        finally:
            import os
            os.unlink(tmp_path)

        results = []
        # GigaAM returns: str, TranscriptionResult, or list
        if isinstance(recognition, str):
            text = recognition.strip()
        elif hasattr(recognition, "text"):
            text = recognition.text.strip()
        elif isinstance(recognition, (list, tuple)):
            text = " ".join(
                item.text.strip() if hasattr(item, "text") else str(item).strip()
                for item in recognition
            ).strip()
        else:
            text = str(recognition).strip()

        if text:
            results.append(TranscriptionResult(
                text=text,
                start=0.0,
                end=len(audio) / sr,
                language="ru",
            ))
        return results

    def unload(self) -> None:
        self._model = None
        log.info("GigaAM model unloaded")
