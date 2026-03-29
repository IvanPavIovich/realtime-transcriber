"""Audio capture from WASAPI loopback or microphone via pyaudiowpatch."""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np

log = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from a WASAPI loopback or microphone into a thread-safe queue."""

    def __init__(self, device_name: str | None, sample_rate: int = 16000,
                 chunk_ms: int = 250, mode: str = "loopback"):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.mode = mode  # "loopback" | "mic"
        self.native_rate: int = sample_rate  # updated after device detection
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream = None
        self._pyaudio = None
        self._stop = threading.Event()

    def _find_device(self):
        """Find WASAPI device by name or auto-detect."""
        import pyaudiowpatch as pyaudio

        self._pyaudio = pyaudio.PyAudio()
        wasapi_info = self._pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)

        is_loopback = self.mode == "loopback"

        target = None
        for i in range(wasapi_info["deviceCount"]):
            dev = self._pyaudio.get_device_info_by_host_api_device_index(
                wasapi_info["index"], i
            )
            if is_loopback:
                if not dev.get("isLoopbackDevice", False):
                    continue
            else:
                # Mic mode: input devices that are NOT loopback
                if dev.get("isLoopbackDevice", False):
                    continue
                if dev.get("maxInputChannels", 0) < 1:
                    continue

            if self.device_name is None:
                target = dev
                break
            if self.device_name.lower() in dev["name"].lower():
                target = dev
                break

        if target is None:
            available = []
            for i in range(wasapi_info["deviceCount"]):
                dev = self._pyaudio.get_device_info_by_host_api_device_index(
                    wasapi_info["index"], i
                )
                if is_loopback and dev.get("isLoopbackDevice", False):
                    available.append(dev["name"])
                elif not is_loopback and not dev.get("isLoopbackDevice", False) and dev.get("maxInputChannels", 0) >= 1:
                    available.append(dev["name"])
            mode_label = "Loopback" if is_loopback else "Microphone"
            raise RuntimeError(
                f"{mode_label} device not found: {self.device_name!r}. "
                f"Available: {available}"
            )
        return target

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback — pushes chunks to queue."""
        import pyaudiowpatch as pyaudio

        if self._stop.is_set():
            return (None, pyaudio.paComplete)

        audio = np.frombuffer(in_data, dtype=np.float32).copy()
        # Mix to mono if stereo
        if len(audio) > frame_count:
            channels = len(audio) // frame_count
            audio = audio.reshape(-1, channels).mean(axis=1)

        try:
            self.queue.put_nowait(audio)
        except queue.Full:
            self.queue.get_nowait()  # drop oldest
            self.queue.put_nowait(audio)

        return (None, pyaudio.paContinue)

    def start(self) -> None:
        """Open audio stream and begin capturing."""
        import pyaudiowpatch as pyaudio

        device = self._find_device()
        native_rate = int(device["defaultSampleRate"])
        self.native_rate = native_rate
        channels = int(device["maxInputChannels"])

        log.info(
            "Capturing: %s @ %dHz, %d ch",
            device["name"], native_rate, channels,
        )

        self._stream = self._pyaudio.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=native_rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=self.chunk_samples,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        log.info("Audio capture started")

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Read one chunk from the queue. Returns None on timeout."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop capturing and release resources."""
        self._stop.set()
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None
        log.info("Audio capture stopped")
