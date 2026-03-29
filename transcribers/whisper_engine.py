"""Whisper transcriber using faster-whisper."""

from __future__ import annotations

import logging

import numpy as np

from transcribers.base import AbstractTranscriber, TranscriptionResult

log = logging.getLogger(__name__)


class WhisperTranscriber(AbstractTranscriber):
    """faster-whisper based transcription engine."""

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str = "float16",
        device: str = "auto",
        language: str = "ru",
        beam_size: int = 5,
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.language = language
        self.beam_size = beam_size
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        log.info(
            "Loading Whisper %s (%s, %s)...",
            self.model_size, self.device, self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device if self.device != "auto" else "cuda",
            compute_type=self.compute_type,
        )
        log.info("Whisper model loaded")

    def transcribe(self, audio: np.ndarray, sr: int) -> list[TranscriptionResult]:
        if self._model is None:
            self.load()

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # We handle VAD ourselves
        )

        # Known Whisper hallucinations on silence/noise
        _hallucinations = {
            "динамичная музыка", "музыка", "субтитры",
            "продолжение следует", "спасибо за просмотр",
            "подписывайтесь на канал", "thanks for watching",
        }

        results = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            # Skip hallucinations
            if text.lower().strip("., !") in _hallucinations:
                log.debug("Skipped hallucination: %s", text)
                continue
            # Skip very low confidence
            if seg.no_speech_prob > 0.8:
                log.debug("Skipped high no_speech_prob: %.2f %s", seg.no_speech_prob, text)
                continue
            results.append(TranscriptionResult(
                text=text,
                start=seg.start,
                end=seg.end,
                language=info.language,
                confidence=seg.avg_logprob,
            ))
        return results

    def unload(self) -> None:
        self._model = None
        log.info("Whisper model unloaded")
