"""Pyannote speaker diarization engine."""

from __future__ import annotations

import logging
import os

import numpy as np

from diarizers.base import AbstractDiarizer, SpeakerSegment

log = logging.getLogger(__name__)


class PyannoteDiarizer(AbstractDiarizer):
    """Pyannote 3.1 speaker diarization."""

    def __init__(
        self,
        min_speakers: int = 1,
        max_speakers: int = 6,
        device: str = "auto",
    ):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self._pipeline = None

    def load(self) -> None:
        from pyannote.audio import Pipeline

        token = os.environ.get("HF_TOKEN", "")
        log.info("Loading Pyannote diarization pipeline...")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token or None,
        )
        if self.device == "cuda":
            import torch
            self._pipeline.to(torch.device("cuda"))
        log.info("Pyannote pipeline loaded")

    def diarize(self, audio: np.ndarray, sr: int) -> list[SpeakerSegment]:
        if self._pipeline is None:
            self.load()

        import torch

        # Pyannote expects (channel, samples) tensor
        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        diarization = self._pipeline(
            {"waveform": waveform, "sample_rate": sr},
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            ))
        return segments

    def unload(self) -> None:
        self._pipeline = None
        log.info("Pyannote pipeline unloaded")
