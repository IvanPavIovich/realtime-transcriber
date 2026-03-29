"""Abstract base for all diarization engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SpeakerSegment:
    """Who spoke when."""
    speaker: str   # e.g. "SPEAKER_00"
    start: float   # seconds
    end: float     # seconds


class AbstractDiarizer(ABC):
    """Interface every diarizer must implement."""

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""

    @abstractmethod
    def diarize(self, audio: np.ndarray, sr: int) -> list[SpeakerSegment]:
        """Run diarization, return speaker segments."""

    @abstractmethod
    def unload(self) -> None:
        """Free model resources."""
