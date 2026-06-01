"""Abstract base class for all audio inference models.

All model wrappers (VAD, KWS, SED, ASC) inherit from this class.
Provides a uniform interface for loading, inference, and backend selection.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceResult:
    """Result from a single model inference call."""

    task: str  # "vad", "kws", "sed", "asc"
    label: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for audio inference models.

    Subclasses must implement:
        - _load_model(): Load the model into memory.
        - _infer(audio): Run inference on audio data.
        - sample_rate: Property returning the expected sample rate.

    The class handles backend selection (ONNX / TensorRT) and provides
    a uniform infer() interface with latency tracking.
    """

    def __init__(
        self,
        model_path: str | Path,
        backend: str = "onnx",
        use_gpu: bool = True,
    ):
        """Initialize model.

        Args:
            model_path: Path to the model file (.onnx or .trt).
            backend: "onnx" or "tensorrt".
            use_gpu: Whether to use GPU acceleration.
        """
        self._model_path = Path(model_path)
        self._backend = backend
        self._use_gpu = use_gpu
        self._model: Any = None
        self._is_loaded = False

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected input sample rate."""
        ...

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Human-readable task name (e.g., 'vad', 'kws')."""
        ...

    def load(self) -> None:
        """Load the model into memory.

        Handles ONNX vs TensorRT backend selection automatically.
        """
        if self._is_loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}. Run `python scripts/download_models.py` first."
            )

        logger.info(f"Loading {self.task_name} model from {self._model_path} (backend={self._backend})")
        self._load_model()
        self._is_loaded = True
        logger.info(f"{self.task_name} model loaded successfully")

    def unload(self) -> None:
        """Unload the model and free resources."""
        self._model = None
        self._is_loaded = False
        logger.debug(f"{self.task_name} model unloaded")

    def infer(self, audio: np.ndarray) -> InferenceResult:
        """Run inference on audio data.

        Args:
            audio: 1-D numpy array of float32 samples at model's sample_rate.

        Returns:
            InferenceResult with label, confidence, and timing.
        """
        if not self._is_loaded:
            raise RuntimeError(f"{self.task_name} model not loaded. Call load() first.")

        t_start = time.perf_counter()
        result = self._infer(audio)
        latency = (time.perf_counter() - t_start) * 1000

        result.task = self.task_name
        result.latency_ms = latency
        result.timestamp = time.time()

        return result

    @abstractmethod
    def _load_model(self) -> None:
        """Platform-specific model loading logic."""
        ...

    @abstractmethod
    def _infer(self, audio: np.ndarray) -> InferenceResult:
        """Platform-specific inference logic."""
        ...

    def validate_audio(self, audio: np.ndarray) -> None:
        """Validate that audio input matches expected format.

        Raises:
            ValueError: If audio format is incorrect.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio array, got shape {audio.shape}. Use audio.squeeze() for mono input.")
        if audio.dtype != np.float32:
            raise ValueError(f"Expected float32 audio, got {audio.dtype}. Use audio.astype(np.float32).")
