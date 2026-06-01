"""Audio capture stream using sounddevice (PortAudio).

Provides a callback-based InputStream that feeds a RingBuffer in real-time.
The capture runs in a high-priority thread managed by PortAudio internally.
"""

import threading
from collections.abc import Callable
from typing import Optional

import numpy as np
import sounddevice as sd

from src.capture.device import AudioDevice, find_device
from src.capture.ring_buffer import RingBuffer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AudioCapture:
    """Manages a real-time audio input stream with callback-based capture.

    Audio is captured in PortAudio's internal high-priority thread and written
    to a lock-free RingBuffer. The consumer reads from the ring buffer in its
    own thread.

    Usage:
        capture = AudioCapture(sample_rate=16000, device="USB")
        capture.start()
        # ... in another thread ...
        audio_chunk = capture.read(480)  # 30ms @ 16kHz
        capture.stop()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str | int | None = None,
        channels: int = 1,
        block_size: int = 480,  # 30ms @ 16kHz
        buffer_duration_sec: float = 3.0,
        dtype: str = "float32",
    ):
        """Initialize audio capture.

        Args:
            sample_rate: Target sample rate in Hz (16000 is standard for speech).
            device: Device name substring, index, or None for default.
            channels: Number of input channels (1 = mono).
            block_size: Samples per callback invocation.
            buffer_duration_sec: Ring buffer capacity in seconds.
            dtype: Sample data type ('float32' or 'int16').
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size
        self._dtype = dtype

        # Resolve device
        self._device_info: Optional[AudioDevice] = find_device(device)
        if self._device_info is None and device is not None:
            logger.warning(f"Device '{device}' not found, using default")

        # Create ring buffer
        buffer_capacity = int(sample_rate * buffer_duration_sec)
        self._ring_buffer = RingBuffer(
            capacity_samples=buffer_capacity,
            channels=channels,
            dtype=np.dtype(dtype),
        )

        # Stream handle
        self._stream: Optional[sd.InputStream] = None
        self._is_running = threading.Event()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        return self._is_running.is_set()

    @property
    def device_name(self) -> str:
        if self._device_info:
            return self._device_info.name
        return "default"

    def start(self) -> None:
        """Start the audio capture stream.

        Raises:
            RuntimeError: If already running.
        """
        if self._is_running.is_set():
            raise RuntimeError("AudioCapture is already running")

        device_idx = self._device_info.index if self._device_info else None

        logger.info(
            f"Starting audio capture: device={self.device_name}, "
            f"sr={self._sample_rate}Hz, block={self._block_size}samples"
        )

        self._stream = sd.InputStream(
            device=device_idx,
            channels=self._channels,
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            dtype=self._dtype,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._is_running.set()
        logger.info("Audio capture started")

    def stop(self) -> None:
        """Stop the audio capture stream."""
        if not self._is_running.is_set():
            return

        self._is_running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio capture stopped")

    def read(self, n_samples: int | None = None) -> np.ndarray:
        """Read captured audio from the ring buffer.

        Args:
            n_samples: Number of samples to read, or None for all available.

        Returns:
            numpy array of shape (n_read, channels).
        """
        return self._ring_buffer.read(n_samples)

    @property
    def available(self) -> int:
        """Number of unread samples in the buffer."""
        return self._ring_buffer.available

    def reset(self) -> None:
        """Clear the ring buffer."""
        self._ring_buffer.reset()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio callback — called from high-priority audio thread.

        Must be fast and never allocate memory or take locks.
        """
        if status:
            logger.debug(f"Audio callback status: {status}")
        self._ring_buffer.write(indata)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class MockAudioCapture(AudioCapture):
    """AudioCapture that generates silence for testing on machines without a mic."""

    def __init__(self, sample_rate: int = 16000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._mock_thread: Optional[threading.Thread] = None
        self._mock_stop = threading.Event()

    def start(self) -> None:
        self._is_running.set()
        self._mock_stop.clear()
        self._mock_thread = threading.Thread(
            target=self._mock_loop,
            daemon=True,
            name="mock-audio-capture",
        )
        self._mock_thread.start()
        logger.info("Mock audio capture started (silence generator)")

    def stop(self) -> None:
        self._is_running.clear()
        self._mock_stop.set()
        if self._mock_thread:
            self._mock_thread.join(timeout=1.0)
            self._mock_thread = None
        logger.info("Mock audio capture stopped")

    def _mock_loop(self) -> None:
        """Generate silence blocks at the configured rate."""
        block = np.zeros((self._block_size, self._channels), dtype=self._dtype)
        sleep_time = self._block_size / self._sample_rate

        while not self._mock_stop.is_set():
            self._ring_buffer.write(block)
            self._mock_stop.wait(sleep_time)
