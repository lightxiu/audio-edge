"""Audio Scene Classification (ASC) model wrapper.

Classifies the acoustic environment (indoor/outdoor, office/street/cafe, etc.)
using a pre-trained audio classification model.

Supports ONNX Runtime inference with mock fallback for development.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from src.models.base import BaseModel, InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

ASC_SAMPLE_RATE = 16000
ASC_DURATION_SEC = 3.0

# Common acoustic scene labels
DEFAULT_SCENE_LABELS = [
    "Indoor",
    "Outdoor",
    "Office",
    "Street",
    "Cafe/Restaurant",
    "Home",
    "Park",
    "Public transport",
    "Car",
    "Market",
    "Library",
    "Station",
]


class MockASC(BaseModel):
    """Mock audio scene classifier for pipeline testing.

    Cycles through scenes with low frequency changes.
    """

    def __init__(self, scenes: Optional[list[str]] = None, **kwargs):
        super().__init__(model_path="mock", backend="mock", use_gpu=False)
        self._scenes = scenes or DEFAULT_SCENE_LABELS[:5]
        self._current_scene = "Indoor"
        self._rng = np.random.RandomState(44)
        self._call_count = 0

    @property
    def sample_rate(self) -> int:
        return ASC_SAMPLE_RATE

    @property
    def task_name(self) -> str:
        return "asc"

    @property
    def scenes(self) -> list[str]:
        return self._scenes

    def load(self) -> None:
        if self._is_loaded:
            return
        self._load_model()
        self._is_loaded = True

    def _load_model(self) -> None:
        logger.info(f"Mock ASC loaded with {len(self._scenes)} scenes")

    def _infer(self, audio: np.ndarray) -> InferenceResult:
        """Mock ASC — stays on same scene, changes ~5% of the time."""
        self._call_count += 1

        # Scene changes rarely (~5%)
        if self._rng.random() < 0.05 and self._call_count > 10:
            self._current_scene = self._rng.choice(self._scenes)
            confidence = 0.6 + self._rng.random() * 0.3
        else:
            confidence = 0.75 + self._rng.random() * 0.2

        return InferenceResult(
            task=self.task_name,
            label=self._current_scene,
            confidence=float(confidence),
            metadata={"scenes": self._scenes},
        )


class ASTSceneClassifier(BaseModel):
    """Audio Spectrogram Transformer (AST) scene classifier.

    AST fine-tuned on AudioSet for acoustic scene classification.
    Expects mel spectrogram input of shape [1, 128, 100] (or similar).

    Model source: HuggingFace — MIT/ast-finetuned-audioset
    License: MIT

    To obtain the ONNX model, export from HuggingFace Transformers:
      from transformers import ASTForAudioClassification
      model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset")
      # Export to ONNX...
    """

    def __init__(
        self,
        model_path: str | Path = "models/asc/ast-finetuned.onnx",
        labels_path: str | Path = "models/asc/scene_labels.txt",
        threshold: float = 0.5,
        min_duration_sec: float = 2.0,
        **kwargs,
    ):
        """Initialize AST scene classifier.

        Args:
            model_path: Path to AST ONNX model.
            labels_path: Path to scene labels (one per line).
            threshold: Minimum confidence to report.
            min_duration_sec: Minimum persistence for scene change.
        """
        super().__init__(model_path=model_path, **kwargs)
        self._labels_path = Path(labels_path)
        self._threshold = threshold
        self._min_duration = min_duration_sec
        self._labels: list[str] = []

    @property
    def sample_rate(self) -> int:
        return ASC_SAMPLE_RATE

    @property
    def task_name(self) -> str:
        return "asc"

    def load(self) -> None:
        """Load AST model and labels."""
        if self._is_loaded:
            return

        if not self._model_path.exists():
            logger.warning(
                f"AST model not found at {self._model_path}. "
                "Falling back to MockASC."
            )
            self._is_loaded = True
            return

        self._load_model()
        self._is_loaded = True

    def _load_model(self) -> None:
        """Load AST ONNX model and scene labels."""
        import onnxruntime as ort

        providers = self._get_providers()
        self._model = ort.InferenceSession(
            str(self._model_path),
            providers=providers,
        )

        # Load scene labels
        if self._labels_path.exists():
            with open(self._labels_path, encoding="utf-8") as f:
                self._labels = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self._labels)} scene labels")
        else:
            self._labels = DEFAULT_SCENE_LABELS
            logger.warning(f"Scene labels not found, using defaults")

        logger.info(f"AST scene classifier loaded: {len(self._labels)} scenes")

    def _infer(self, audio: np.ndarray) -> InferenceResult:
        """Run ASC inference.

        Args:
            audio: 1-D float32 array at 16kHz, ~3s recommended.

        Returns:
            InferenceResult with scene label.
        """
        self.validate_audio(audio)

        # Fallback to mock if model not loaded
        if self._model is None:
            mock = MockASC(scenes=self._labels if self._labels else None)
            mock._is_loaded = True
            return mock._infer(audio)

        # Compute mel spectrogram (AST expects 128 mel bins, ~100 frames)
        from src.features.mel import compute_mel_spectrogram

        mel = compute_mel_spectrogram(
            audio,
            sample_rate=ASC_SAMPLE_RATE,
            n_mels=128,
            n_fft=400,
            hop_length=160,
        )

        # Ensure fixed size (truncate or pad)
        target_frames = 100
        features = mel.features
        if features.shape[1] < target_frames:
            pad = target_frames - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad)), mode="constant")
        elif features.shape[1] > target_frames:
            features = features[:, :target_frames]

        # Run ONNX inference
        input_tensor = features.astype(np.float32).reshape(1, 128, target_frames)
        input_name = self._model.get_inputs()[0].name
        scores = self._model.run(None, {input_name: input_tensor})[0].squeeze()

        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])

        if top_score >= self._threshold and top_idx < len(self._labels):
            label = self._labels[top_idx]
        else:
            label = "Unknown"

        return InferenceResult(
            task=self.task_name,
            label=label,
            confidence=top_score,
            metadata={
                "scene_index": top_idx,
                "threshold": self._threshold,
            },
        )

    def _get_providers(self) -> list[str]:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if self._use_gpu and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
