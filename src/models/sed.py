"""Sound Event Detection (SED) model wrapper.

YAMNet-based sound event detection with 521 AudioSet classes.
Supports ONNX Runtime inference with mock fallback for development.

YAMNet: https://tfhub.dev/google/yamnet/1
License: Apache 2.0
"""

from pathlib import Path

import numpy as np

from src.models.base import BaseModel, InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

SED_SAMPLE_RATE = 16000
SED_DURATION_SEC = 0.96  # YAMNet window: 96 frames × 10ms hop

# Top AudioSet classes we care about (subset of 521)
DEFAULT_SED_CLASSES = [
    "Speech",
    "Silence",
    "Siren",
    "Gunshot, gunfire",
    "Glass break",
    "Dog bark",
    "Car horn",
    "Fire alarm",
    "Police car (siren)",
    "Ambulance (siren)",
    "Baby cry, infant cry",
    "Explosion",
    "Screaming",
    "Thunder",
    "Door slam",
    "Engine",
    "Alarm clock",
    "Telephone ring",
    "Knock",
    "Water",
]


class MockSED(BaseModel):
    """Mock sound event detector for pipeline testing.

    Returns random environmental sound events at low frequency.
    """

    def __init__(self, classes: list[str] | None = None, **kwargs):
        super().__init__(model_path="mock", backend="mock", use_gpu=False)
        self._classes = classes or DEFAULT_SED_CLASSES[:10]
        self._rng = np.random.RandomState(43)

    @property
    def sample_rate(self) -> int:
        return SED_SAMPLE_RATE

    @property
    def task_name(self) -> str:
        return "sed"

    @property
    def classes(self) -> list[str]:
        return self._classes

    def load(self) -> None:
        if self._is_loaded:
            return
        self._load_model()
        self._is_loaded = True

    def _load_model(self) -> None:
        logger.info(f"Mock SED loaded with {len(self._classes)} classes")

    def _infer(self, audio: np.ndarray) -> InferenceResult:
        """Mock SED — occasionally returns a random sound event."""
        # ~3% chance of detecting something
        if self._rng.random() < 0.03:
            event = self._rng.choice(self._classes)
            confidence = 0.3 + self._rng.random() * 0.6
        else:
            event = "Silence"
            confidence = 0.7 + self._rng.random() * 0.25

        return InferenceResult(
            task=self.task_name,
            label=event,
            confidence=float(confidence),
            metadata={"classes": self._classes},
        )


class YAMNetSED(BaseModel):
    """YAMNet Sound Event Detection ONNX wrapper.

    YAMNet is a MobileNet-based audio classification model trained on
    AudioSet with 521 sound event classes.

    The ONNX model expects:
      - Input: mel spectrogram of shape [1, 96, 64] (96 frames × 64 mel bands)
      - Output: scores [1, 521], embeddings [1, 1024]

    To obtain the ONNX model, export from TensorFlow Hub:
      https://tfhub.dev/google/yamnet/1
    """

    def __init__(
        self,
        model_path: str | Path = "models/sed/yamnet.onnx",
        labels_path: str | Path = "models/sed/yamnet_class_map.csv",
        threshold: float = 0.3,
        target_classes: list[str] | None = None,
        **kwargs,
    ):
        """Initialize YAMNet SED.

        Args:
            model_path: Path to YAMNet ONNX model.
            labels_path: Path to CSV class map (id, display_name).
            threshold: Minimum confidence to report an event.
            target_classes: List of class names to detect (None = all).
        """
        super().__init__(model_path=model_path, **kwargs)
        self._labels_path = Path(labels_path)
        self._threshold = threshold
        self._target_classes = target_classes or []
        self._labels: dict[int, str] = {}  # index → class name

    @property
    def sample_rate(self) -> int:
        return SED_SAMPLE_RATE

    @property
    def task_name(self) -> str:
        return "sed"

    @property
    def threshold(self) -> float:
        return self._threshold

    def load(self) -> None:
        """Load YAMNet model and label map."""
        if self._is_loaded:
            return

        if not self._model_path.exists():
            logger.warning(
                f"YAMNet model not found at {self._model_path}. "
                "Falling back to MockSED. Export YAMNet from TF Hub to use real model."
            )
            self._is_loaded = True
            return

        self._load_model()
        self._is_loaded = True

    def _load_model(self) -> None:
        """Load YAMNet ONNX model and class labels."""
        import csv

        import onnxruntime as ort

        # Load ONNX model
        providers = self._get_providers()
        self._model = ort.InferenceSession(
            str(self._model_path),
            providers=providers,
        )

        # Load class labels from CSV
        if self._labels_path.exists():
            with open(self._labels_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        idx = int(row[0])
                        name = row[2].strip()
                        self._labels[idx] = name
            logger.info(f"Loaded {len(self._labels)} YAMNet class labels")

        # Filter to target classes
        if self._target_classes:
            self._target_indices = [idx for idx, name in self._labels.items() if name in self._target_classes]
            logger.info(f"Filtered to {len(self._target_indices)} target classes")

        logger.info(f"YAMNet SED loaded: {len(self._labels)} classes")

    def _infer(self, audio: np.ndarray) -> InferenceResult:
        """Run SED inference.

        Args:
            audio: 1-D float32 array at 16kHz, 0.96s recommended.

        Returns:
            InferenceResult with top sound event or "Silence".
        """
        self.validate_audio(audio)

        # Fallback to mock if model not loaded
        if self._model is None:
            mock = MockSED(classes=list(self._labels.values()) if self._labels else None)
            mock._is_loaded = True
            return mock._infer(audio)

        # Compute mel spectrogram onnx input
        mel_spec = self._extract_yamnet_mel(audio)
        mel_spec = mel_spec.astype(np.float32).reshape(1, 96, 64)

        # Run ONNX inference
        input_name = self._model.get_inputs()[0].name
        scores, embeddings = self._model.run(None, {input_name: mel_spec})
        scores = scores.squeeze()  # [521]

        # Find top class
        if self._target_classes:
            # Only consider target classes
            indices = self._target_indices
            if not indices:
                indices = range(len(scores))
        else:
            indices = range(len(scores))

        top_idx = max(indices, key=lambda i: scores[i])
        top_score = float(scores[top_idx])

        if top_score >= self._threshold:
            label = self._labels.get(top_idx, f"class_{top_idx}")
        else:
            label = "Silence"
            top_score = 1.0 - top_score

        return InferenceResult(
            task=self.task_name,
            label=label,
            confidence=top_score,
            metadata={
                "class_index": top_idx,
                "threshold": self._threshold,
            },
        )

    @staticmethod
    def _extract_yamnet_mel(audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram matching YAMNet's expected input.

        YAMNet expects: 64 mel bands, 96 frames (0.96s @ 10ms hop),
        sample rate 16kHz, STFT window 25ms, hop 10ms, Hann window.
        """
        from src.features.mel import compute_mel_spectrogram

        mel = compute_mel_spectrogram(
            audio,
            sample_rate=SED_SAMPLE_RATE,
            n_mels=64,
            n_fft=400,  # 25ms @ 16kHz
            hop_length=160,  # 10ms @ 16kHz
            f_min=125.0,
            f_max=7500.0,
        )

        # Ensure exactly 96 frames
        if mel.features.shape[1] < 96:
            pad = 96 - mel.features.shape[1]
            mel.features = np.pad(mel.features, ((0, 0), (0, pad)), mode="constant")
        elif mel.features.shape[1] > 96:
            mel.features = mel.features[:, :96]

        return mel.features

    def _get_providers(self) -> list[str]:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if self._use_gpu and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
