# Realtime Transcriber

Транскрипция аудио в реальном времени с определением спикеров. Захватывает звук с микрофона или системного вывода (WASAPI loopback), детектирует речь через Silero VAD, транскрибирует через Whisper или GigaAM, определяет спикеров через Pyannote.

## Возможности

- **Захват аудио**: Микрофон или WASAPI loopback (VB-Cable, виртуальные кабели)
- **VAD**: Silero VAD + энергетический фильтр — точная детекция речи/тишины
- **Транскрипция**:
  - faster-whisper (tiny → large-v3)
  - GigaAM v3 (Сбер) — на 50% точнее Whisper на русском, в 6 раз легче (240M vs 1.5B параметров)
- **Диаризация**: Pyannote 3.1 — определение спикеров
- **Вывод**: JSONL с таймкодами, ID спикеров, вывод в консоль в реальном времени
- **Фильтр галлюцинаций**: Блокирует известные артефакты Whisper ("ДИНАМИЧНАЯ МУЗЫКА" и т.д.)

## Быстрый старт

```bash
# Установка зависимостей
pip install pyaudiowpatch pyyaml click numpy

# PyTorch (CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyTorch (GPU — CUDA 12.x)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Whisper
pip install faster-whisper

# GigaAM (опционально):
pip install git+https://github.com/salute-developers/GigaAM.git

# Pyannote для диаризации (опционально):
pip install pyannote.audio
```

### Запуск

```bash
# Микрофон, CPU, tiny модель (быстрый тест)
python main.py -c config-dev.yaml

# Системный звук (VB-Cable), GPU, large модель (продакшн)
python main.py -c config.yaml

# Свой конфиг
python main.py -c my-config.yaml
```

Остановка: `Ctrl+C`

---

## Справочник параметров

Все параметры задаются в YAML-конфиге. Скопируйте `config.yaml` или `config-dev.yaml` и настройте под себя.

### audio — Захват аудио

```yaml
audio:
  device_name: null          # Фильтр по имени устройства (null = автоопределение)
  mode: "mic"                # "mic" — микрофон, "loopback" — системный звук
  sample_rate: 16000         # Целевая частота дискретизации
  chunk_ms: 250              # Размер чанка в мс
```

| Параметр | Значения | Описание |
|----------|----------|----------|
| `device_name` | `null` / `"CABLE Output"` / `"Realtek"` | Подстрока имени устройства. `null` — берёт первое доступное. |
| `mode` | `"mic"` / `"loopback"` | **mic** — захват с микрофона. **loopback** — захват системного звука (то, что слышно в колонках). Для loopback нужен VB-Cable или аналог. |
| `sample_rate` | `16000` | Целевая частота. Устройство может работать на своей частоте (например 48000), ресемплинг автоматический. |
| `chunk_ms` | `200`–`500` | Размер чанка. Меньше = быстрее реакция, больше = меньше нагрузка на CPU. По умолчанию `250`. |

**Режим loopback** захватывает всё, что играет через колонки — YouTube, Zoom, игры. Требует виртуальный аудиокабель (VB-Cable).

### vad — Детекция голосовой активности

```yaml
vad:
  threshold: 0.3
  min_speech_ms: 200
  min_silence_ms: 600
  max_segment_sec: 30.0
  padding_ms: 300
```

| Параметр | Диапазон | Описание |
|----------|----------|----------|
| `threshold` | `0.05`–`0.95` | Чувствительность VAD. **Ниже = ловит больше речи** (но больше ложных срабатываний). **Выше = строже** (может пропустить тихую речь). |
| `min_speech_ms` | `100`–`500` | Минимальная длительность речи для начала сегмента. Фильтрует щелчки и шумы. |
| `min_silence_ms` | `300`–`1500` | Сколько ждать тишины перед отрезкой сегмента. **Больше = длиннее сегменты** (больше контекста для транскрипции, точнее). **Меньше = быстрее отклик** (но сегменты могут быть слишком короткими). |
| `max_segment_sec` | `10`–`60` | Принудительная отрезка сегментов длиннее этого значения. |
| `padding_ms` | `100`–`500` | Захват аудио до начала речи. Ловит начала слов, которые VAD мог пропустить. |

#### Настройка под сценарий

| Сценарий | threshold | min_silence_ms | min_speech_ms |
|----------|-----------|----------------|---------------|
| **Тихая комната, близкий микрофон** | `0.3`–`0.5` | `500`–`800` | `200` |
| **Шумная комната, далёкий микрофон** | `0.1`–`0.2` | `800`–`1200` | `150` |
| **Системный звук (loopback)** | `0.4`–`0.6` | `400`–`600` | `250` |
| **Запись совещания** | `0.3` | `600`–`1000` | `200` |
| **Максимальная чувствительность** | `0.05`–`0.15` | `1000` | `100` |

**Речь пропускается** → понизить `threshold` (попробовать `0.15`), увеличить `padding_ms` (`300`–`500`).
**Слишком много шума** → повысить `threshold` (`0.5`+), повысить `min_speech_ms` (`300`+).
**Слова обрезаются на середине** → увеличить `min_silence_ms` (`800`–`1200`).
**Слишком медленный отклик** → уменьшить `min_silence_ms` (`300`–`500`).

### transcriber — Движок транскрипции

```yaml
transcriber:
  engine: "whisper"              # "whisper" | "gigaam"

  whisper:
    model_size: "large-v3"       # Размер модели (см. таблицу)
    compute_type: "float16"      # "float16" (GPU) | "int8" (CPU)
    language: "ru"               # Код языка
    beam_size: 5                 # Ширина beam search (1 = быстрее, 5 = точнее)

  gigaam:
    model_type: "ctc"            # "ctc" | "rnnt"
```

#### Модели Whisper

| Модель | Размер | VRAM | Скорость (CPU) | Скорость (GPU) | Точность | Для чего |
|--------|--------|------|----------------|----------------|----------|----------|
| `tiny` | 75 МБ | ~1 ГБ | ~Реальное время | ⚡ | Низкая | Быстрый тест, слабое железо |
| `base` | 142 МБ | ~1 ГБ | Медленно | ⚡ | Средняя | Лёгкое использование |
| `small` | 466 МБ | ~2 ГБ | Очень медленно | Быстро | Хорошая | Баланс |
| `medium` | 1.5 ГБ | ~5 ГБ | ❌ Слишком медленно | Хорошо | Высокая | Хороший GPU |
| `large-v3` | 3 ГБ | ~8 ГБ | ❌ Слишком медленно | Хорошо | Лучшая | Продакшн (нужен GPU) |

**compute_type:**
- `float16` — только GPU, самый быстрый, лучшее качество
- `int8` — работает на CPU, в 2 раза меньше памяти, чуть ниже качество
- `float32` — максимальная точность, в 2 раза больше памяти чем float16

**beam_size:**
- `1` — жадный декодинг, быстрее всего, ниже точность
- `3` — хороший баланс
- `5` — лучшая точность (по умолчанию для продакшна)

**language:** ISO-код (`"ru"`, `"en"`, `"de"`, `"fr"` и т.д.). Явная установка убирает задержку на автодетекцию.

#### GigaAM (Сбер)

Модель распознавания речи для русского языка. **На 50% точнее Whisper large-v3 на русском** при 6x меньшем количестве параметров.

| Модель | Параметры | Описание |
|--------|-----------|----------|
| `ctc` | 240M | Самый быстрый, подходит для стриминга |
| `rnnt` | 240M | Лучшая точность, чуть медленнее |

```bash
# Установка GigaAM
pip install git+https://github.com/salute-developers/GigaAM.git
```

**Когда использовать GigaAM vs Whisper:**
- Только русский → **GigaAM** (точнее, легче)
- Несколько языков → **Whisper**
- Слабое железо (CPU) → **GigaAM CTC** или **Whisper tiny**
- Лучшее качество на русском → **GigaAM RNNT**

### diarizer — Определение спикеров

```yaml
diarizer:
  enabled: true                  # true | false
  engine: "pyannote"

  pyannote:
    min_speakers: 1              # Минимум ожидаемых спикеров
    max_speakers: 6              # Максимум ожидаемых спикеров
```

Требуется переменная окружения `HF_TOKEN` с токеном Hugging Face (принять лицензию на https://huggingface.co/pyannote/speaker-diarization-3.1).

| Параметр | Описание |
|----------|----------|
| `enabled` | `false` — отключить диаризацию (быстрее, меньше памяти) |
| `min_speakers` | Подсказка по минимальному числу спикеров |
| `max_speakers` | Подсказка по максимальному числу спикеров |

**Примечание:** Диаризация добавляет ~2-5 секунд на сегмент (GPU). На CPU может быть очень медленной. Отключайте для транскрипции одного спикера.

### device — Вычислительное устройство

```yaml
device: "auto"                   # "auto" | "cpu" | "cuda"
```

- `auto` — CUDA если доступна, иначе CPU
- `cpu` — принудительно CPU (медленнее, но работает везде)
- `cuda` — принудительно GPU (нужна NVIDIA GPU + CUDA)

### output — Формат вывода

```yaml
output:
  format: "jsonl"
  file: null                     # Путь к файлу или null (только консоль)
  realtime_print: true           # Вывод в консоль в реальном времени
```

| Параметр | Описание |
|----------|----------|
| `file` | Путь к файлу вывода (напр. `"output/result.jsonl"`). `null` = без файла, только консоль. |
| `realtime_print` | Показывать транскрипцию в консоли по мере обработки. |

---

## Примеры конфигов

### Микрофон на слабом CPU (тест/демо)
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

### Системный звук на GPU (продакшн)
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

### GigaAM для русского (GPU, лучшее качество)
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

### Запись совещания в файл
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

## Архитектура

```
Аудио вход ──► AudioCapture (WASAPI микрофон/loopback)
                    │
                    ▼
              SileroVAD (энергетическая + нейронная детекция речи)
                    │
                    ▼
             SegmentBuffer (накопление речи, отрезка по тишине)
                    │
                    ▼
              Transcriber (Whisper / GigaAM)
                    │
                    ▼
              Diarizer (Pyannote) [опционально]
                    │
                    ▼
             OutputWriter (JSONL + консоль)
```

Ключевые решения:
- **Модульные движки** — новые транскрайберы/диаризаторы добавляются через `AbstractTranscriber` / `AbstractDiarizer`
- **Автоматический ресемплинг** — устройство работает на нативной частоте (напр. 48kHz), звук пересэмплируется в 16kHz для моделей
- **Фильтр галлюцинаций** — блокирует известные артефакты Whisper на тишине

## Формат вывода (JSONL)

```json
{"start": 0.0, "end": 3.5, "text": "Привет, как дела?", "speaker": "SPEAKER_00"}
{"start": 3.8, "end": 7.2, "text": "Всё хорошо, спасибо!", "speaker": "SPEAKER_01"}
```

Без диаризации (один спикер):
```json
{"start": 0.0, "end": 3.5, "text": "Привет, как дела?"}
```

## Системные требования

| Конфигурация | CPU | RAM | GPU VRAM | Модели |
|--------------|-----|-----|----------|--------|
| **Минимум (тест)** | Любой 2+ ядра | 4 ГБ | — | Whisper tiny, без диаризации |
| **Рекомендуемый (CPU)** | 4+ ядра | 8 ГБ | — | Whisper small / GigaAM CTC |
| **Продакшн (GPU)** | 4+ ядра | 16 ГБ | 8+ ГБ | Whisper large-v3 / GigaAM RNNT + Pyannote |
| **Максимальное качество** | 6+ ядер | 32 ГБ | 16+ ГБ | GigaAM RNNT + Pyannote, beam_size 5 |

**ОС:** Windows 10/11 (WASAPI loopback). Linux/macOS — только режим микрофона (для loopback нужен PulseAudio/BlackHole).

## Решение проблем

| Проблема | Решение |
|----------|---------|
| Речь не детектируется | Понизить `vad.threshold` (попробовать `0.1`–`0.2`), увеличить `padding_ms` |
| Слишком много шума | Повысить `vad.threshold` (`0.5`+), повысить `min_speech_ms` |
| Слова обрезаются посреди фразы | Увеличить `min_silence_ms` (`800`–`1200`) |
| "ДИНАМИЧНАЯ МУЗЫКА" в выводе | Встроенный фильтр галлюцинаций. Для новых — добавить в `_hallucinations` в `whisper_engine.py` |
| Медленно на CPU | Использовать `tiny` или `base`, `beam_size: 1`, отключить диаризацию |
| Нет loopback-устройств | Установить [VB-Cable](https://vb-audio.com/Cable/) или использовать `mode: "mic"` |
| Не хватает памяти GPU | Использовать меньшую модель или `compute_type: "int8"` |
| Ошибка авторизации Pyannote | Установить переменную `HF_TOKEN` с токеном Hugging Face |
