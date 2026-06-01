"""Tests for the lock-free ring buffer."""

import numpy as np

from src.capture.ring_buffer import RingBuffer


class TestRingBuffer:
    """Unit tests for RingBuffer."""

    def test_initial_state(self):
        buf = RingBuffer(capacity_samples=1000, channels=1)
        assert buf.capacity == 1000
        assert buf.channels == 1
        assert buf.available == 0

    def test_write_read_simple(self):
        buf = RingBuffer(capacity_samples=1000, channels=1)
        data = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

        buf.write(data)
        assert buf.available == 3

        result = buf.read(3)
        np.testing.assert_array_almost_equal(result, data)

    def test_write_read_partial(self):
        buf = RingBuffer(capacity_samples=1000, channels=1)
        data = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)

        buf.write(data)
        result = buf.read(2)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result, data[:2])
        assert buf.available == 3

    def test_read_all_available(self):
        buf = RingBuffer(capacity_samples=1000, channels=1)
        data = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

        buf.write(data)
        result = buf.read()  # Read all
        assert len(result) == 3
        assert buf.available == 0

    def test_read_more_than_available(self):
        buf = RingBuffer(capacity_samples=1000, channels=1)
        buf.write(np.array([[0.1]], dtype=np.float32))

        result = buf.read(100)
        assert len(result) == 1  # Only 1 available
        assert buf.available == 0

    def test_read_empty_buffer(self):
        buf = RingBuffer(capacity_samples=1000, channels=1)
        result = buf.read(10)
        assert len(result) == 0
        assert result.shape == (0, 1)

    def test_wrap_around(self):
        """Test that data wraps correctly around the buffer boundary."""
        buf = RingBuffer(capacity_samples=5, channels=1)

        # Fill buffer
        buf.write(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32))
        assert buf.available == 5

        # Read first 3, leaving 2 at the end
        buf.read(3)
        assert buf.available == 2

        # Write 3 more — should wrap
        buf.write(np.array([[6.0], [7.0], [8.0]], dtype=np.float32))
        assert buf.available == 5

        # Read all — should get correct sequence
        result = buf.read(5)
        expected = np.array([[4.0], [5.0], [6.0], [7.0], [8.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_write_larger_than_capacity(self):
        """Test that writing more than capacity only keeps the tail."""
        buf = RingBuffer(capacity_samples=3, channels=1)
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)

        buf.write(data)
        # Should only keep last 3
        result = buf.read(3)
        expected = np.array([[3.0], [4.0], [5.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_reset(self):
        buf = RingBuffer(capacity_samples=100, channels=1)
        buf.write(np.array([[1.0], [2.0]], dtype=np.float32))
        assert buf.available == 2

        buf.reset()
        assert buf.available == 0
        result = buf.read(10)
        assert len(result) == 0

    def test_multi_channel(self):
        buf = RingBuffer(capacity_samples=100, channels=2)
        data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        buf.write(data)
        result = buf.read(2)
        np.testing.assert_array_almost_equal(result, data)

    def test_overwrite_old_data(self):
        """When consumer falls behind, old data is silently dropped."""
        buf = RingBuffer(capacity_samples=5, channels=1)

        # Write 5 samples
        buf.write(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32))
        # Don't read — write 5 more (should overwrite)
        buf.write(np.array([[6.0], [7.0], [8.0], [9.0], [10.0]], dtype=np.float32))

        # Should only have last 5
        result = buf.read(10)
        expected = np.array([[6.0], [7.0], [8.0], [9.0], [10.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
