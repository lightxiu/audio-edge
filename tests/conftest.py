"""Test fixtures and configuration for pytest."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def silence_audio(sample_rate: int):
    """Generate 1 second of silence."""
    import numpy as np

    return np.zeros((sample_rate, 1), dtype=np.float32)


@pytest.fixture
def sine_audio(sample_rate: int):
    """Generate 1 second of 440 Hz sine wave."""
    import numpy as np

    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return audio.reshape(-1, 1)
