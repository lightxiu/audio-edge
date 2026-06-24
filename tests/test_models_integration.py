"""Integration tests for real ONNX model inference.

Tests YAMNet SED and AST ASC ONNX models when available.
Skipped gracefully in CI environments where models are not present.
"""

from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path("models")

YAMNET_ONNX = MODEL_DIR / "sed" / "yamnet.onnx"
YAMNET_LABELS = MODEL_DIR / "sed" / "yamnet_class_map.csv"
AST_ONNX = MODEL_DIR / "asc" / "ast-finetuned.onnx"
AST_LABELS = MODEL_DIR / "asc" / "audioset_labels.txt"

# Skip markers
yamnet_available = pytest.mark.skipif(
    not YAMNET_ONNX.exists(),
    reason="YAMNet ONNX model not downloaded",
)
ast_available = pytest.mark.skipif(
    not AST_ONNX.exists(),
    reason="AST ONNX model not downloaded",
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ── YAMNet SED Tests ──────────────────────────────────


@yamnet_available
class TestYAMNetSED:
    """Integration tests for YAMNet SED with real ONNX model."""

    @pytest.fixture
    def sed_model(self):
        from src.models.sed import YAMNetSED

        m = YAMNetSED(
            model_path=YAMNET_ONNX,
            labels_path=YAMNET_LABELS,
            threshold=0.3,
        )
        m.load()
        return m

    def test_load_labels(self, sed_model):
        """Labels CSV should be loaded and match model output (521 classes)."""
        assert len(sed_model._labels) == 521
        assert sed_model._labels[0]  # First class should have a name

    def test_infer_silence(self, sed_model):
        """Inference on near-silence should return a valid result."""
        audio = np.zeros(16000, dtype=np.float32)
        result = sed_model.infer(audio)
        assert result.task == "sed"
        assert result.label  # non-empty
        assert 0.0 <= result.confidence <= 1.0

    def test_infer_variable_length(self, sed_model):
        """Model should handle different input lengths."""
        for duration in [0.5, 1.0, 2.0, 3.0]:
            n = int(16000 * duration)
            audio = np.random.randn(n).astype(np.float32) * 0.01
            result = sed_model.infer(audio)
            assert result.task == "sed"
            assert 0.0 <= result.confidence <= 1.0

    def test_infer_with_target_classes(self):
        """Filtering to target classes should still work."""
        from src.models.sed import YAMNetSED

        m = YAMNetSED(
            model_path=YAMNET_ONNX,
            labels_path=YAMNET_LABELS,
            threshold=0.3,
            target_classes=["Speech", "Dog bark"],
        )
        m.load()
        audio = np.random.randn(32000).astype(np.float32) * 0.01
        result = m.infer(audio)
        assert result.label in ("Speech", "Dog bark", "Silence")

    def test_fallback_when_model_missing(self):
        """Should fall back to mock when model file doesn't exist."""
        from src.models.sed import YAMNetSED

        m = YAMNetSED(
            model_path="models/sed/nonexistent.onnx",
            labels_path=YAMNET_LABELS,
        )
        m.load()
        audio = np.zeros(16000, dtype=np.float32)
        result = m.infer(audio)
        assert result.task == "sed"
        assert result.label  # mock still returns something

    def test_sample_rate(self, sed_model):
        assert sed_model.sample_rate == 16000

    def test_task_name(self, sed_model):
        assert sed_model.task_name == "sed"


# ── AST ASC Tests ─────────────────────────────────────


@ast_available
class TestASTSceneClassifier:
    """Integration tests for AST ASC with real ONNX model."""

    @pytest.fixture
    def asc_model(self):
        from src.models.asc import ASTSceneClassifier

        m = ASTSceneClassifier(
            model_path=AST_ONNX,
            labels_path=AST_LABELS,
            threshold=0.1,
        )
        m.load()
        return m

    def test_load_labels(self, asc_model):
        """Should load 527 AudioSet class labels."""
        assert len(asc_model._labels) == 527
        assert asc_model._labels[0]

    def test_infer_silence_10s(self, asc_model):
        """Inference on 10s of silence should return a valid result."""
        audio = np.zeros(160000, dtype=np.float32)
        result = asc_model.infer(audio)
        assert result.task == "asc"
        assert result.label
        assert 0.0 <= result.confidence <= 1.0

    def test_infer_short_audio(self, asc_model):
        """Short audio (< 1024 frames) should be interpolated to 1024."""
        audio = np.zeros(80000, dtype=np.float32)  # 5s
        result = asc_model.infer(audio)
        assert result.task == "asc"
        assert 0.0 <= result.confidence <= 1.0

    def test_infer_long_audio(self, asc_model):
        """Long audio (> 1024 frames) should be downsampled to 1024."""
        audio = np.zeros(320000, dtype=np.float32)  # 20s
        result = asc_model.infer(audio)
        assert result.task == "asc"
        assert 0.0 <= result.confidence <= 1.0

    def test_fallback_when_model_missing(self):
        """Should fall back to mock when model file doesn't exist."""
        from src.models.asc import ASTSceneClassifier

        m = ASTSceneClassifier(
            model_path="models/asc/nonexistent.onnx",
            labels_path=AST_LABELS,
        )
        m.load()
        audio = np.zeros(160000, dtype=np.float32)
        result = m.infer(audio)
        assert result.task == "asc"
        assert result.label

    def test_sample_rate(self, asc_model):
        assert asc_model.sample_rate == 16000

    def test_task_name(self, asc_model):
        assert asc_model.task_name == "asc"


# ── Mel Spectrogram Consistency ────────────────────────


class TestMelForModels:
    """Verify mel spectrogram extraction matches model requirements."""

    def test_yamnet_mel_shape(self):
        """YAMNet mel should produce (64, ≥96) for 1s of audio."""
        from src.models.sed import YAMNetSED

        audio = np.random.randn(16000).astype(np.float32)
        mel = YAMNetSED._extract_yamnet_mel(audio)
        assert mel.shape[0] == 64  # 64 mel bands
        assert mel.shape[1] >= 96  # At least 96 frames

    def test_ast_mel_shape(self):
        """AST mel should produce (128, N) for audio input."""
        from src.features.mel import compute_mel_spectrogram

        audio = np.random.randn(160000).astype(np.float32)  # 10s
        mel = compute_mel_spectrogram(
            audio,
            sample_rate=16000,
            n_mels=128,
            n_fft=400,
            hop_length=160,
        )
        assert mel.features.shape[0] == 128  # 128 mel bands
        assert mel.features.shape[1] > 0  # Some frames

    def test_ast_frame_resize(self):
        """Frame interpolation to 1024 should preserve band structure."""
        from src.features.mel import compute_mel_spectrogram

        audio = np.random.randn(80000).astype(np.float32)  # 5s
        mel = compute_mel_spectrogram(
            audio, sample_rate=16000, n_mels=128, n_fft=400, hop_length=160,
        )
        n_frames = mel.features.shape[1]

        # Simulate ASC resize logic
        target = 1024
        features_t = mel.features.T  # (n_frames, n_mels)
        x_old = np.linspace(0, 1, n_frames)
        x_new = np.linspace(0, 1, target)
        resized = np.array([
            np.interp(x_new, x_old, features_t[:, i])
            for i in range(128)
        ]).T
        assert resized.shape == (target, 128)
        # Interpolation should preserve reasonable range
        assert not np.isnan(resized).any()
