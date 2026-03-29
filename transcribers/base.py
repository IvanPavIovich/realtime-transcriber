"""Abstract base for all transcription engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionResult:
    """Single transcription segment."""
    text: str
    start: float  # seconds
    end: float    # seconds
    language: str = ""
    confidence: float = 0.0


class AbstractTranscriber(ABC):
    """Interface every transcriber must implement."""

    @abstractmethod
    def load(self) -> None:
        """Load model into memory. Called once at startup."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int) -> list[TranscriptionResult]:
        """Transcribe audio array, return list of segments."""

    @abstractmethod
    def unload(self) -> None:
        """Free model resources."""
