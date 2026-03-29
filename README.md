# Realtime Transcriber

Real-time audio transcription with speaker diarization. Captures audio from microphone or system output (WASAPI loopback), detects speech via Silero VAD, transcribes with Whisper or GigaAM, identifies speakers with Pyannote.

## Features

- **Audio Capture**: Microphone or WASAPI loopback (VB-Cable, virtual cables)
- **VAD**: Silero VAD + energy filter — accurate speech/silence detection
- **Transcription**:
  - faster-whisper (tiny → large-v3)
  - GigaAM v3 (Sber) — 50% more accurate than Whisper on Russian, 6x smaller model (240M vs 1.5B params)
- **Diarization**: Pyannote 3.1 — speaker identification
- **Output**: JSONL with timestamps, speaker IDs, real-time console output
- **Hallucination filter**: Blocks known Whisper artifacts ("ДИНАМИЧНАЯ МУЗЫКА", etc.)

## Quick Start

```bash
# Install core dependencies
pip install pyaudiowpatch pyyaml click numpy

# Install PyTorch (CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch (GPU — CUDA 12.x)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Whisper
pip install faster-whisper

# Install GigaAM (optional):
pip install git+https://github.com/salute-developers/GigaAM.git

# Install Pyannote for diarization (optional):
pip install pyannote.audio
```

### Run

```bash
# Microphone, CPU, tiny model (quick test)
python main.py -c config-dev.yaml

# System audio (VB-Cable), GPU, large model (production)
python main.py -c config.yaml

# Custom config
python main.py -c my-config.yaml
```

Press `Ctrl+C` to stop.

---

## Configuration Reference

All parameters are in YAML config files. Copy `config.yaml` or `config-dev.yaml` and adjust.

### audio — Audio Capture

```yaml
audio:
  device_name: null          # Device name filter (null = auto-detect first available)
  mode: "mic"                # "mic" — microphone, "loopback" — system audio (WASAPI)
  sample_rate: 16000         # Target sample rate (resampling is automatic)
  chunk_ms: 250              # Chunk size in ms (how often audio is read)
```

| Parameter | Values | Description |
|-----------|--------|-------------|
| `device_name` | `null` / `"CABLE Output"` / `"Realtek"` | Substring match against device name. `null` picks the first available device. |
| `mode` | `"mic"` / `"loopback"` | **mic** — captures from microphone. **loopback** — captures system audio output (what you hear in speakers). Loopback requires WASAPI-compatible device (VB-Cable, etc.). |
| `sample_rate` | `16000` | Internal target rate. Device may run at its native rate (e.g. 48000Hz), resampling is handled automatically. |
| `chunk_ms` | `200`–`500` | Size of each audio chunk. Lower = more responsive, higher = less CPU. Default `250` is good. |

**Loopback mode** captures everything playing through speakers — YouTube, Zoom calls, games. Requires a virtual audio cable (VB-Cable) or WASAPI loopback-capable device.

### vad — Voice Activity Detection

```yaml
vad:
  threshold: 0.3
  min_speech_ms: 200
  min_silence_ms: 600
  max_segment_sec: 30.0
  padding_ms: 300
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| `threshold` | `0.05`–`0.95` | VAD sensitivity. **Lower = catches more speech** (but more false positives). **Higher = stricter** (may miss quiet speech). |
| `min_speech_ms` | `100`–`500` | Minimum speech duration to start a segment. Filters out clicks/noise. |
| `min_silence_ms` | `300`–`1500` | How long to wait for silence before cutting a segment. **Higher = longer segments** (more context for transcription, more accurate). **Lower = faster response** (but segments may be too short for Whisper). |
| `max_segment_sec` | `10`–`60` | Force-cut segments longer than this. Prevents memory issues on continuous speech. |
| `padding_ms` | `100`–`500` | Audio padding before speech start. Captures the beginning of words that VAD might miss. |

#### Tuning Guide

| Scenario | threshold | min_silence_ms | min_speech_ms |
|----------|-----------|----------------|---------------|
| **Quiet room, close mic** | `0.3`–`0.5` | `500`–`800` | `200` |
| **Noisy room, far mic** | `0.1`–`0.2` | `800`–`1200` | `150` |
| **System audio (loopback)** | `0.4`–`0.6` | `400`–`600` | `250` |
| **Meeting recording** | `0.3` | `600`–`1000` | `200` |
| **Maximum sensitivity** | `0.05`–`0.15` | `1000` | `100` |

**If speech is missed** → lower `threshold` (try `0.15`), increase `padding_ms` (try `300`–`500`).
**If too much noise triggers** → raise `threshold` (try `0.5`+), raise `min_speech_ms` (try `300`+).
**If words are cut in the middle** → increase `min_silence_ms` (try `800`–`1200`).
**If response is too slow** → decrease `min_silence_ms` (try `300`–`500`).

### transcriber — Speech Recognition Engine

```yaml
transcriber:
  engine: "whisper"              # "whisper" | "gigaam"

  whisper:
    model_size: "large-v3"       # Model size (see table below)
    compute_type: "float16"      # "float16" (GPU) | "int8" (CPU)
    language: "ru"               # Language code
    beam_size: 5                 # Beam search width (1 = fastest, 5 = most accurate)

  gigaam:
    model_type: "ctc"            # "ctc" | "rnnt"
```

#### Whisper Models

| Model | Size | VRAM | Speed (CPU) | Speed (GPU) | Accuracy | Best for |
|-------|------|------|-------------|-------------|----------|----------|
| `tiny` | 75 MB | ~1 GB | ~Real-time | ⚡ | Low | Quick test, weak hardware |
| `base` | 142 MB | ~1 GB | Slow | ⚡ | Medium | Light usage |
| `small` | 466 MB | ~2 GB | Very slow | Fast | Good | Balanced |
| `medium` | 1.5 GB | ~5 GB | ❌ Too slow | Good | High | Good GPU |
| `large-v3` | 3 GB | ~8 GB | ❌ Too slow | Good | Best | Production (GPU required) |

**compute_type:**
- `float16` — GPU only, fastest, best quality
- `int8` — CPU compatible, 2x smaller memory, slightly lower quality
- `float32` — maximum precision, 2x more memory than float16

**beam_size:**
- `1` — greedy decoding, fastest, lowest accuracy
- `3` — good balance
- `5` — best accuracy (default for production)

**language:** ISO code (`"ru"`, `"en"`, `"de"`, `"fr"`, etc.). Setting explicitly avoids auto-detection delay.

#### GigaAM (Sber)

Russian-language ASR model. **50% more accurate than Whisper large-v3 on Russian** with 6x fewer parameters.

| Model | Params | Description |
|-------|--------|-------------|
| `ctc` | 240M | Fastest, streaming-friendly |
| `rnnt` | 240M | Better accuracy, slightly slower |

```bash
# Install GigaAM
pip install git+https://github.com/salute-developers/GigaAM.git
```

**When to use GigaAM vs Whisper:**
- Russian only → **GigaAM** (more accurate, lighter)
- Multiple languages → **Whisper**
- Weak hardware (CPU) → **GigaAM CTC** or **Whisper tiny**
- Best Russian quality → **GigaAM RNNT**

### diarizer — Speaker Identification

```yaml
diarizer:
  enabled: true                  # true | false
  engine: "pyannote"

  pyannote:
    min_speakers: 1              # Minimum expected speakers
    max_speakers: 6              # Maximum expected speakers
```

Requires `HF_TOKEN` environment variable with a Hugging Face token (accept license at https://huggingface.co/pyannote/speaker-diarization-3.1).

| Parameter | Description |
|-----------|-------------|
| `enabled` | `false` to skip diarization (faster, less memory) |
| `min_speakers` | Hint for minimum speaker count |
| `max_speakers` | Hint for maximum speaker count |

**Note:** Diarization adds ~2-5s processing time per segment on GPU. On CPU it may be very slow. Disable for simple single-speaker transcription.

### device — Compute Device

```yaml
device: "auto"                   # "auto" | "cpu" | "cuda"
```

- `auto` — uses CUDA if available, falls back to CPU
- `cpu` — force CPU (slower, but works everywhere)
- `cuda` — force GPU (requires NVIDIA GPU + CUDA toolkit)

### output — Output Format

```yaml
output:
  format: "jsonl"                # "jsonl"
  file: null                     # File path or null (stdout only)
  realtime_print: true           # Print to console in real-time
```

| Parameter | Description |
|-----------|-------------|
| `file` | Path to output file (e.g. `"output/result.jsonl"`). `null` = no file, console only. |
| `realtime_print` | Show transcription in console as it happens. |

---

## Example Configs

### Microphone on weak CPU (test/demo)
```yaml
audio:
  mode: "mic"
  device_name: null
vad:
  threshold: 0.15
  min_silence_ms: 800
transcriber:
  engine: "whisper"
  whisper:
    model_size: "tiny"
    compute_type: "int8"
    beam_size: 1
diarizer:
  enabled: false
device: "cpu"
```

### System audio on GPU (production)
```yaml
audio:
  mode: "loopback"
  device_name: "CABLE Output"
vad:
  threshold: 0.5
  min_silence_ms: 500
transcriber:
  engine: "whisper"
  whisper:
    model_size: "large-v3"
    compute_type: "float16"
    beam_size: 5
diarizer:
  enabled: true
  engine: "pyannote"
  pyannote:
    max_speakers: 6
device: "cuda"
```

### GigaAM for Russian (GPU, best quality)
```yaml
audio:
  mode: "loopback"
  device_name: "CABLE Output"
vad:
  threshold: 0.4
  min_silence_ms: 600
transcriber:
  engine: "gigaam"
  gigaam:
    model_type: "rnnt"
diarizer:
  enabled: true
device: "cuda"
```

### Meeting recording to file
```yaml
audio:
  mode: "loopback"
  device_name: null
vad:
  threshold: 0.3
  min_silence_ms: 1000
  max_segment_sec: 60
transcriber:
  engine: "whisper"
  whisper:
    model_size: "medium"
    compute_type: "float16"
    beam_size: 5
diarizer:
  enabled: true
  pyannote:
    min_speakers: 2
    max_speakers: 4
device: "cuda"
output:
  file: "output/meeting.jsonl"
  realtime_print: true
```

---

## Architecture

```
Audio Input ──► AudioCapture (WASAPI mic/loopback)
                    │
                    ▼
              SileroVAD (energy + neural speech detection)
                    │
                    ▼
             SegmentBuffer (accumulate speech, cut on silence)
                    │
                    ▼
              Transcriber (Whisper / GigaAM)
                    │
                    ▼
              Diarizer (Pyannote) [optional]
                    │
                    ▼
             OutputWriter (JSONL + console)
```

Key design:
- **Modular engines** — add new transcribers/diarizers by implementing `AbstractTranscriber` / `AbstractDiarizer`
- **Automatic resampling** — device runs at native rate (e.g. 48kHz), audio is resampled to 16kHz for models
- **Hallucination filter** — blocks known Whisper artifacts on silence

## Output Format (JSONL)

```json
{"start": 0.0, "end": 3.5, "text": "Привет, как дела?", "speaker": "SPEAKER_00"}
{"start": 3.8, "end": 7.2, "text": "Всё хорошо, спасибо!", "speaker": "SPEAKER_01"}
```

Without diarization (single speaker):
```json
{"start": 0.0, "end": 3.5, "text": "Привет, как дела?"}
```

## System Requirements

| Setup | CPU | RAM | GPU VRAM | Models |
|-------|-----|-----|----------|--------|
| **Minimum (test)** | Any 2+ cores | 4 GB | — | Whisper tiny, no diarization |
| **Recommended (CPU)** | 4+ cores | 8 GB | — | Whisper small / GigaAM CTC |
| **Production (GPU)** | 4+ cores | 16 GB | 8+ GB | Whisper large-v3 / GigaAM RNNT + Pyannote |
| **Maximum quality** | 6+ cores | 32 GB | 16+ GB | GigaAM RNNT + Pyannote, beam_size 5 |

**OS:** Windows 10/11 (WASAPI loopback). Linux/macOS — mic mode only (loopback requires PulseAudio/BlackHole).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Speech not detected | Lower `vad.threshold` (try `0.1`–`0.2`), increase `padding_ms` |
| Too much noise triggers | Raise `vad.threshold` (try `0.5`+), raise `min_speech_ms` |
| Words cut mid-sentence | Increase `min_silence_ms` (try `800`–`1200`) |
| "ДИНАМИЧНАЯ МУЗЫКА" output | Built-in hallucination filter handles this. If new hallucinations appear, add to `_hallucinations` set in `whisper_engine.py` |
| Slow on CPU | Use `tiny` or `base` model, `beam_size: 1`, disable diarization |
| No loopback devices | Install [VB-Cable](https://vb-audio.com/Cable/) or use `mode: "mic"` |
| Out of GPU memory | Use smaller model or `compute_type: "int8"` |
| Pyannote auth error | Set `HF_TOKEN` env variable with Hugging Face token |
