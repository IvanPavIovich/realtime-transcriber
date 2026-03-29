# Realtime Transcriber

Real-time audio transcription with speaker diarization. Captures system audio (VB-Cable/WASAPI loopback), detects speech, transcribes with Whisper or GigaAM, identifies speakers with Pyannote.

## Features

- **Audio Capture**: WASAPI loopback via pyaudiowpatch (VB-Cable, virtual cables)
- **VAD**: Silero VAD — accurate speech/silence detection
- **Transcription**:
  - faster-whisper (tiny → large-v3)
  - GigaAM v3 (Sber) — 50% more accurate than Whisper on Russian, 6x smaller model
- **Diarization**: Pyannote 3.1 — speaker identification
- **Output**: JSONL with timestamps, speaker IDs, real-time console output

## Quick Start

```bash
# Install
pip install -r requirements.txt

# For GigaAM (optional):
pip install git+https://github.com/salute-developers/GigaAM.git

# Run with default config (GPU, large-v3)
python main.py

# Run with dev config (CPU, tiny model)
python main.py -c config-dev.yaml
```

## Config

Edit `config.yaml`:

```yaml
transcriber:
  engine: "whisper"    # whisper | gigaam
  whisper:
    model_size: "large-v3"
    compute_type: "float16"

diarizer:
  enabled: true
  engine: "pyannote"

device: "auto"         # auto | cpu | cuda
```

## Architecture

```
Audio Input (VB-Cable) → AudioCapture (WASAPI)
    → SileroVAD (speech detection)
    → SegmentBuffer (accumulate speech)
    → Transcriber (Whisper / GigaAM)
    → Diarizer (Pyannote)
    → OutputWriter (JSONL + console)
```

## Output Format (JSONL)

```json
{"start": 0.0, "end": 3.5, "text": "Привет, как дела?", "speaker": "SPEAKER_00"}
{"start": 3.8, "end": 7.2, "text": "Всё хорошо, спасибо!", "speaker": "SPEAKER_01"}
```

## Requirements

- Python 3.10+
- Windows (WASAPI loopback) — VB-Cable or similar virtual audio cable
- GPU recommended for large models (VRAM 8GB+)
- CPU works with tiny/base/small models or GigaAM CTC
