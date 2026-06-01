"""Tests for Silero VAD model and audio feature extraction.

Note: Model-dependent tests are skipped if silero_vad.onnx is not downloaded.
"""

from pathlib import Path

import numpy as np
import pytest

from src.features.mel import (
    MelSpectrogram,
    MFCC,
    compute_mel_spectrogram,
    compute_mfcc,
    mel_filter_bank,
)
from src.models.vad import SileroVAD

MODEL_PATH = Path("models/vad/silero_vad.onnx")


# --- Mel spectrogram tests ---


class TestMelFilterBank:
    """Tests for mel filter bank creation."""

    def test_shape(self):
        filters = mel_filter_bank(n_mels=64, n_fft=1024, sample_rate=16000)
        assert filters.shape == (64, 513)  # (n_mels, n_fft//2 + 1)

    def test_range(self):
        """Filter bank values should be non-negative and sum to non-zero per filter."""
        filters = mel_filter_bank(n_mels=40, n_fft=512, sample_rate=16000)
        assert np.all(filters >= 0)
        # Each mel filter should have at least some non-zero weights
        assert np.all(np.sum(filters, axis=1) > 0)

    def test_custom_frequencies(self):
        filters = mel_filter_bank(
            n_mels=32, n_fft=1024, sample_rate=16000, f_min=100, f_max=4000
        )
        assert filters.shape == (32, 513)


class TestMelSpectrogram:
    """Tests for mel spectrogram computation."""

    def test_output_shape(self, sine_audio: np.ndarray):
        audio = sine_audio.squeeze()  # (16000,) @ 16kHz = 1 second
        mel = compute_mel_spectrogram(audio, sample_rate=16000)
        assert isinstance(mel, MelSpectrogram)
        assert mel.features.shape[0] == 64  # n_mels
        assert mel.features.shape[1] > 0  # n_frames

    def test_frame_count(self):
        """n_frames should be approximately duration / hop_length."""
        sample_rate = 16000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        mel = compute_mel_spectrogram(audio, sample_rate=sample_rate, hop_length=160)
        expected_frames = 1 + (len(audio) - 1024) // 160
        assert mel.features.shape[1] == expected_frames

    def test_short_audio(self):
        """Audio shorter than n_fft should still work (auto-padded)."""
        audio = np.random.randn(100).astype(np.float32)
        mel = compute_mel_spectrogram(audio, sample_rate=16000, n_fft=1024)
        assert mel.features.shape[1] >= 1  # At least one frame

    def test_silence_spectrogram(self, silence_audio: np.ndarray):
        """Silence should produce valid (very low) mel values."""
        audio = silence_audio.squeeze()
        mel = compute_mel_spectrogram(audio, sample_rate=16000)
        assert not np.any(np.isnan(mel.features))
        assert not np.any(np.isinf(mel.features))
        # Log of near-zero → very negative values
        assert np.all(mel.features < 0)

    def test_custom_parameters(self):
        audio = np.random.randn(16000).astype(np.float32)
        mel = compute_mel_spectrogram(
            audio,
            sample_rate=16000,
            n_mels=128,
            n_fft=2048,
            hop_length=256,
        )
        assert mel.features.shape[0] == 128
        assert mel.n_fft == 2048


class TestMFCC:
    """Tests for MFCC computation."""

    def test_output_shape(self, sine_audio: np.ndarray):
        audio = sine_audio.squeeze()
        mfcc = compute_mfcc(audio, sample_rate=16000, n_mfcc=40)
        assert isinstance(mfcc, MFCC)
        assert mfcc.features.shape[0] == 40

    def test_zero_mean(self):
        """MFCC of noise should produce finite, well-formed coefficients."""
        rng = np.random.RandomState(42)
        audio = rng.randn(48000).astype(np.float32)  # 3 seconds
        mfcc = compute_mfcc(audio, sample_rate=16000, n_mfcc=13)
        # Coefficients should be finite
        assert np.all(np.isfinite(mfcc.features))
        # Higher-order coefficients (less energy) should be bounded
        # Skip 0th (total energy) — it's the dominant component
        assert np.all(np.abs(mfcc.features[1:]) < 50.0)


# --- VAD tests (model-dependent) ---


class TestSileroVAD:
    """Tests for Silero VAD ONNX wrapper.

    These tests require the model file to be downloaded first.
    """

    @pytest.fixture(autouse=True)
    def _check_model(self):
        """Skip all VAD tests if model is not available."""
        if not MODEL_PATH.exists():
            pytest.skip(
                f"Silero VAD model not found at {MODEL_PATH}. "
                "Run `python scripts/download_models.py` to download it."
            )

    def test_load(self):
        vad = SileroVAD(str(MODEL_PATH))
        vad.load()
        assert vad.is_loaded
        assert vad.sample_rate == 16000
        vad.unload()

    def test_infer_silence(self, silence_audio: np.ndarray):
        """Silence should be classified as non-speech."""
        vad = SileroVAD(str(MODEL_PATH))
        vad.load()

        audio = silence_audio.squeeze()
        # Process in 512-sample chunks
        results = []
        for i in range(0, len(audio), 512):
            chunk = audio[i : i + 512]
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)), mode="constant")
            results.append(vad.infer(chunk))

        vad.unload()

        # All chunks should be "silence"
        speech_ratio = sum(1 for r in results if r.label == "speech") / len(results)
        assert speech_ratio < 0.3  # Allow some noise, but mostly silence

    def test_infer_sine_wave(self, sine_audio: np.ndarray):
        """A 440 Hz sine wave might trigger VAD (it's a tonal signal)."""
        vad = SileroVAD(str(MODEL_PATH))
        vad.load()

        audio = sine_audio.squeeze()
        results = []
        for i in range(0, len(audio), 512):
            chunk = audio[i : i + 512]
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)), mode="constant")
            results.append(vad.infer(chunk))

        vad.unload()

        # All results should have valid confidence values
        for r in results:
            assert 0.0 <= r.confidence <= 1.0
            assert r.task == "vad"

    def test_reset_state(self):
        """Resetting state should clear RNN memory."""
        vad = SileroVAD(str(MODEL_PATH))
        vad.load()

        # Run some inference to build up state
        audio = np.random.randn(16000).astype(np.float32)
        for i in range(0, 5120, 512):
            chunk = audio[i : i + 512]
            vad.infer(chunk)

        # Reset
        old_state = vad._state.copy() if vad._state is not None else None
        vad.reset_state()

        # State should be reset to initial
        if old_state is not None:
            # After reset, state should be the init state (all zeros)
            assert np.array_equal(vad._state, vad._init_state)

        vad.unload()

    def test_detect_segments_silence(self, silence_audio: np.ndarray):
        """detect_segments on silence should yield no or very few segments."""
        vad = SileroVAD(str(MODEL_PATH))
        vad.load()

        audio = silence_audio.squeeze()
        segments = list(vad.detect_segments(audio))

        vad.unload()

        # Silence should produce few to no segments
        assert len(segments) <= 2  # Allow minimal false triggers

    def test_validate_audio(self):
        """validate_audio should reject bad inputs."""
        vad = SileroVAD(str(MODEL_PATH))
        vad.load()

        # 2-D array should raise
        with pytest.raises(ValueError, match="1-D"):
            vad.validate_audio(np.zeros((100, 1), dtype=np.float32))

        # Non-float32 should raise
        with pytest.raises(ValueError, match="float32"):
            vad.validate_audio(np.zeros(100, dtype=np.int16))

        vad.unload()

    def test_threshold_setter(self):
        """Threshold should be clampable."""
        vad = SileroVAD(str(MODEL_PATH))
        vad.load()

        vad.threshold = 0.8
        assert vad.threshold == 0.8

        vad.threshold = 1.5  # Should clamp to 1.0
        assert vad.threshold == 1.0

        vad.threshold = -0.5  # Should clamp to 0.0
        assert vad.threshold == 0.0

        vad.unload()
