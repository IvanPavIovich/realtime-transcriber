"""Main transcription pipeline — orchestrates capture → VAD → buffer → transcribe → diarize → output."""

from __future__ import annotations

import logging
import time

import numpy as np

from core.audio_capture import AudioCapture
from core.output import OutputWriter
from core.segment_buffer import SegmentBuffer
from core.vad import SileroVAD
from diarizers.base import AbstractDiarizer, SpeakerSegment
from transcribers.base import AbstractTranscriber, TranscriptionResult

log = logging.getLogger(__name__)


def _assign_speakers(
    results: list[TranscriptionResult],
    speakers: list[SpeakerSegment],
    offset: float,
) -> list[tuple[TranscriptionResult, str | None]]:
    """Match transcription segments to speaker labels by time overlap."""
    if not speakers:
        return [(r, None) for r in results]

    paired = []
    for r in results:
        abs_start = offset + r.start
        abs_end = offset + r.end
        best_speaker = None
        best_overlap = 0.0

        for s in speakers:
            overlap_start = max(abs_start, s.start)
            overlap_end = min(abs_end, s.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s.speaker

        paired.append((r, best_speaker))
    return paired


class TranscriptionPipeline:
    """Full pipeline: audio → VAD → segment → transcribe → diarize → output."""

    def __init__(
        self,
        capture: AudioCapture,
        vad: SileroVAD,
        buffer: SegmentBuffer,
        transcriber: AbstractTranscriber,
        diarizer: AbstractDiarizer | None,
        output: OutputWriter,
        sample_rate: int = 16000,
    ):
        self.capture = capture
        self.vad = vad
        self.buffer = buffer
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.output = output
        self.sample_rate = sample_rate
        self._running = False
        self._segment_count = 0
        self._start_time: float = 0.0

    def start(self) -> None:
        """Initialize all components."""
        log.info("Starting pipeline...")
        self.vad.load()
        self.transcriber.load()
        if self.diarizer:
            self.diarizer.load()
        self.capture.start()
        # Use device native sample rate for VAD
        self.vad.sample_rate = self.capture.native_rate
        self._running = True
        self._start_time = time.time()
        log.info("Pipeline running. Press Ctrl+C to stop.")

    def run(self) -> None:
        """Main loop — blocks until stopped."""
        self.start()
        try:
            while self._running:
                chunk = self.capture.read(timeout=1.0)
                if chunk is None:
                    continue
                self._process_chunk(chunk)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            self.stop()

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """VAD → buffer → transcribe on segment boundary."""
        speech = self.vad.is_speech(chunk)
        segment = self.buffer.push(chunk, speech)

        if segment is not None:
            self._handle_segment(segment)

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Resample audio if sample rates differ."""
        if from_sr == to_sr:
            return audio
        import torchaudio
        import torch
        tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, from_sr, to_sr)
        return resampled.squeeze(0).numpy()

    def _handle_segment(self, audio: np.ndarray) -> None:
        """Transcribe (and optionally diarize) a complete segment."""
        self._segment_count += 1
        native_sr = self.capture.native_rate
        offset = time.time() - self._start_time - len(audio) / native_sr

        # Resample to 16kHz for transcription
        audio_16k = self._resample(audio, native_sr, 16000)

        # Transcribe
        results = self.transcriber.transcribe(audio_16k, 16000)
        if not results:
            return

        # Diarize
        speakers: list[SpeakerSegment] = []
        if self.diarizer:
            speakers = self.diarizer.diarize(audio, self.sample_rate)

        # Match speakers to transcription segments
        paired = _assign_speakers(results, speakers, offset)

        # Output
        for result, speaker in paired:
            # Adjust timestamps to absolute
            result.start += offset
            result.end += offset
            self.output.write(result, speaker)

    def stop(self) -> None:
        """Flush and shutdown."""
        self._running = False

        # Flush remaining audio
        remaining = self.buffer.flush()
        if remaining is not None:
            self._handle_segment(remaining)

        self.capture.stop()
        self.vad.reset()
        self.output.close()
        log.info(
            "Pipeline stopped. Processed %d segments in %.0fs",
            self._segment_count, time.time() - self._start_time,
        )
