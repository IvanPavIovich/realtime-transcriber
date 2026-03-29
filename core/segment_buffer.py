"""Accumulates audio chunks into speech segments based on VAD decisions."""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


class SegmentBuffer:
    """Collects speech chunks, emits complete segments on silence."""

    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        max_segment_sec: float = 30.0,
        padding_ms: int = 200,
    ):
        self.sample_rate = sample_rate
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self.max_segment_samples = int(sample_rate * max_segment_sec)
        self.padding_samples = int(sample_rate * padding_ms / 1000)

        self._chunks: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._segment_start: float | None = None
        self._padding_buffer: list[np.ndarray] = []

    def push(self, chunk: np.ndarray, is_speech: bool) -> np.ndarray | None:
        """Add a chunk. Returns a complete segment when ready, else None.

        Args:
            chunk: Audio samples (float32, mono).
            is_speech: VAD decision for this chunk.

        Returns:
            Concatenated audio segment if a boundary is detected, else None.
        """
        n = len(chunk)

        if is_speech:
            if self._segment_start is None:
                self._segment_start = time.time()
                # Prepend padding from before speech started
                for pad in self._padding_buffer:
                    self._chunks.append(pad)

            self._chunks.append(chunk)
            self._speech_samples += n
            self._silence_samples = 0
            self._padding_buffer.clear()

            # Force-emit on max length
            if self._speech_samples >= self.max_segment_samples:
                return self._emit()
        else:
            if self._chunks:
                # In a segment — accumulate silence
                self._silence_samples += n
                self._chunks.append(chunk)

                if self._silence_samples >= self.min_silence_samples:
                    if self._speech_samples >= self.min_speech_samples:
                        return self._emit()
                    else:
                        self._reset()
            else:
                # Not in a segment — keep rolling padding buffer
                self._padding_buffer.append(chunk)
                total_pad = sum(len(c) for c in self._padding_buffer)
                while total_pad > self.padding_samples and self._padding_buffer:
                    removed = self._padding_buffer.pop(0)
                    total_pad -= len(removed)

        return None

    def flush(self) -> np.ndarray | None:
        """Force-emit whatever is buffered (call at shutdown)."""
        if self._chunks and self._speech_samples >= self.min_speech_samples:
            return self._emit()
        self._reset()
        return None

    def _emit(self) -> np.ndarray:
        segment = np.concatenate(self._chunks)
        log.debug("Segment emitted: %.1fs", len(segment) / self.sample_rate)
        self._reset()
        return segment

    def _reset(self) -> None:
        self._chunks.clear()
        self._speech_samples = 0
        self._silence_samples = 0
        self._segment_start = None
        self._padding_buffer.clear()
