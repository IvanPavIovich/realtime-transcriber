"""Realtime Transcriber — entry point.

Usage:
    python main.py                      # default config.yaml
    python main.py -c config-dev.yaml   # dev/test config
"""

from __future__ import annotations

import logging
import sys

import click
import yaml

from core.audio_capture import AudioCapture
from core.output import OutputWriter
from core.pipeline import TranscriptionPipeline
from core.segment_buffer import SegmentBuffer
from core.vad import SileroVAD

log = logging.getLogger("transcriber")


def _resolve_device(cfg_device: str) -> str:
    """Resolve 'auto' to actual device string."""
    if cfg_device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def _build_transcriber(cfg: dict, device: str):
    """Factory: create transcriber from config."""
    engine = cfg["transcriber"]["engine"]

    if engine == "whisper":
        from transcribers.whisper_engine import WhisperTranscriber
        wcfg = cfg["transcriber"]["whisper"]
        compute = wcfg.get("compute_type", "float16" if device == "cuda" else "int8")
        return WhisperTranscriber(
            model_size=wcfg["model_size"],
            compute_type=compute,
            device=device,
            language=wcfg.get("language", "ru"),
            beam_size=wcfg.get("beam_size", 5),
        )
    elif engine == "gigaam":
        from transcribers.gigaam_engine import GigaAMTranscriber
        gcfg = cfg["transcriber"].get("gigaam", {})
        return GigaAMTranscriber(
            model_type=gcfg.get("model_type", "ctc"),
            device=device,
        )
    else:
        raise ValueError(f"Unknown transcriber engine: {engine}")


def _build_diarizer(cfg: dict, device: str):
    """Factory: create diarizer from config (or None if disabled)."""
    dcfg = cfg.get("diarizer", {})
    if not dcfg.get("enabled", False):
        return None

    engine = dcfg.get("engine", "pyannote")

    if engine == "pyannote":
        from diarizers.pyannote_engine import PyannoteDiarizer
        pcfg = dcfg.get("pyannote", {})
        return PyannoteDiarizer(
            min_speakers=pcfg.get("min_speakers", 1),
            max_speakers=pcfg.get("max_speakers", 6),
            device=device,
        )
    else:
        raise ValueError(f"Unknown diarizer engine: {engine}")


@click.command()
@click.option("-c", "--config", "config_path", default="config.yaml", help="Path to config YAML")
def main(config_path: str):
    """Real-time audio transcription with speaker diarization."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Load config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = _resolve_device(cfg.get("device", "auto"))
    log.info("Device: %s", device)

    # Audio capture
    acfg = cfg["audio"]
    capture = AudioCapture(
        device_name=acfg.get("device_name"),
        sample_rate=acfg.get("sample_rate", 16000),
        chunk_ms=acfg.get("chunk_ms", 250),
        mode=acfg.get("mode", "loopback"),
    )

    # VAD
    vcfg = cfg["vad"]
    vad = SileroVAD(
        threshold=vcfg.get("threshold", 0.5),
        min_speech_ms=vcfg.get("min_speech_ms", 250),
        min_silence_ms=vcfg.get("min_silence_ms", 500),
        sample_rate=acfg.get("sample_rate", 16000),
    )

    # Segment buffer
    buffer = SegmentBuffer(
        sample_rate=acfg.get("sample_rate", 16000),
        min_speech_ms=vcfg.get("min_speech_ms", 250),
        min_silence_ms=vcfg.get("min_silence_ms", 500),
        max_segment_sec=vcfg.get("max_segment_sec", 30.0),
        padding_ms=vcfg.get("padding_ms", 200),
    )

    # Transcriber
    transcriber = _build_transcriber(cfg, device)

    # Diarizer
    diarizer = _build_diarizer(cfg, device)

    # Output
    ocfg = cfg.get("output", {})
    output = OutputWriter(
        file_path=ocfg.get("file"),
        realtime_print=ocfg.get("realtime_print", True),
    )

    # Pipeline
    pipeline = TranscriptionPipeline(
        capture=capture,
        vad=vad,
        buffer=buffer,
        transcriber=transcriber,
        diarizer=diarizer,
        output=output,
        sample_rate=acfg.get("sample_rate", 16000),
    )

    pipeline.run()


if __name__ == "__main__":
    main()
