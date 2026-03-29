"""Output formatters — JSONL to stdout or file."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from transcribers.base import TranscriptionResult

log = logging.getLogger(__name__)


class OutputWriter:
    """Writes transcription results to stdout or file in JSONL format."""

    def __init__(
        self,
        file_path: str | None = None,
        realtime_print: bool = True,
    ):
        self.realtime_print = realtime_print
        self._file = None

        if file_path:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(p, "a", encoding="utf-8")
            log.info("Output file: %s", p)

    def write(
        self,
        result: TranscriptionResult,
        speaker: str | None = None,
    ) -> None:
        """Write one transcription result."""
        record = {
            "start": round(result.start, 3),
            "end": round(result.end, 3),
            "text": result.text,
        }
        if speaker:
            record["speaker"] = speaker
        if result.language:
            record["language"] = result.language

        line = json.dumps(record, ensure_ascii=False)

        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

        if self.realtime_print:
            prefix = f"[{speaker}] " if speaker else ""
            ts = f"{result.start:.1f}-{result.end:.1f}s"
            print(f"  {ts}  {prefix}{result.text}", flush=True)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
