"""Lock-free single-producer single-consumer (SPSC) ring buffer for real-time audio.

Uses a pre-allocated numpy array with atomic write/read pointers.
The capture thread writes; the VAD thread reads. No locks needed.
"""

import numpy as np


class RingBuffer:
    """Lock-free SPSC ring buffer backed by a numpy array.

    The producer (audio capture callback) calls write().
    The consumer (VAD / preprocessing thread) calls read().

    Capacity is fixed at creation time. If the producer outruns the consumer,
    old data is silently overwritten (no blocking).
    """

    def __init__(self, capacity_samples: int, channels: int = 1, dtype: np.dtype = np.float32):
        """Initialize ring buffer.

        Args:
            capacity_samples: Maximum number of samples per channel to store.
            channels: Number of audio channels (1 for mono).
            dtype: numpy dtype for audio samples.
        """
        self._capacity = capacity_samples
        self._channels = channels
        self._dtype = dtype

        # Pre-allocate storage
        self._buffer = np.zeros((capacity_samples, channels), dtype=dtype)

        # Write index (producer advances this)
        self._write_idx: int = 0
        # Total samples written (monotonic counter, never wraps)
        self._total_written: int = 0
        # Total samples read (monotonic counter, never wraps)
        self._total_read: int = 0

    @property
    def capacity(self) -> int:
        """Buffer capacity in samples."""
        return self._capacity

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels

    @property
    def available(self) -> int:
        """Number of unread samples available."""
        return self._total_written - self._total_read

    def write(self, data: np.ndarray) -> None:
        """Write audio samples to the buffer (called from capture callback).

        Args:
            data: numpy array of shape (n_samples, channels).
        """
        n = len(data)
        if n > self._capacity:
            # Input larger than buffer — only keep the tail
            data = data[-self._capacity :]
            n = self._capacity

        # Write to buffer with wrap-around
        idx = self._write_idx
        end = idx + n

        if end <= self._capacity:
            self._buffer[idx:end] = data
        else:
            # Wraps around
            first_part = self._capacity - idx
            self._buffer[idx:] = data[:first_part]
            self._buffer[: end - self._capacity] = data[first_part:]

        self._write_idx = (self._write_idx + n) % self._capacity
        self._total_written += n

        # If consumer is falling behind, advance read pointer to avoid
        # reading very stale data (keep at most capacity samples)
        if self.available > self._capacity:
            self._total_read = self._total_written - self._capacity

    def read(self, n_samples: int | None = None) -> np.ndarray:
        """Read audio samples from the buffer (called from consumer thread).

        Args:
            n_samples: Number of samples to read. If None, read all available.
                       If more than available, returns what's available.

        Returns:
            numpy array of shape (n_read, channels). May be empty if no data.
        """
        if n_samples is None:
            n_samples = self.available

        n = min(n_samples, self.available)
        if n == 0:
            return np.zeros((0, self._channels), dtype=self._dtype)

        idx = self._total_read % self._capacity
        end = idx + n

        if end <= self._capacity:
            result = self._buffer[idx:end].copy()
        else:
            # Wraps around
            first_part = self._capacity - idx
            result = np.concatenate(
                [self._buffer[idx:], self._buffer[: end - self._capacity]],
                axis=0,
            )

        self._total_read += n
        return result

    def reset(self) -> None:
        """Reset the buffer to empty state."""
        self._write_idx = 0
        self._total_written = 0
        self._total_read = 0
        self._buffer.fill(0)
