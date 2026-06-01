"""Keyword Spotting (KWS) model wrapper.

Supports two backends:
  - sherpa-onnx: Full-featured KWS with Zipformer models (recommended)
  - Mock KWS: For development/testing without a real model

sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
License: Apache 2.0
"""

from pathlib import Path
from typing import Optional

import numpy as np

from src.models.base import BaseModel, InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

KWS_SAMPLE_RATE = 16000


class MockKWS(BaseModel):
    """Mock keyword spotter for testing the pipeline without a real model.

    Always returns "no_keyword" with random confidence. Useful for
    developing and testing the orchestrator on machines without sherpa-onnx.
    """

    def __init__(self, keywords: Optional[list[str]] = None, **kwargs):
        super().__init__(model_path="mock", backend="mock", use_gpu=False)
        self._keywords = keywords or ["hey_computer", "stop", "go"]
        self._rng = np.random.RandomState(42)

    @property
    def sample_rate(self) -> int:
        return KWS_SAMPLE_RATE

    @property
    def task_name(self) -> str:
        return "kws"

    @property
    def keywords(self) -> list[str]:
        return self._keywords

    def load(self) -> None:
        """Load mock KWS (no file needed)."""
        if self._is_loaded:
            return
        self._load_model()
        self._is_loaded = True

    def _load_model(self) -> None:
        logger.info(f"Mock KWS loaded with keywords: {self._keywords}")

    def _infer(self, audio: np.ndarray) -> InferenceResult:
        """Mock inference — always returns no_keyword or rarely a random keyword."""
        # Simulate very rare keyword detection (~2% chance)
        if self._rng.random() < 0.02:
            keyword = self._rng.choice(self._keywords)
            confidence = 0.5 + self._rng.random() * 0.45
        else:
            keyword = "no_keyword"
            confidence = self._rng.random() * 0.3

        return InferenceResult(
            task=self.task_name,
            label=keyword,
            confidence=float(confidence),
            metadata={"keywords": self._keywords},
        )


class SherpaKWS(BaseModel):
    """sherpa-onnx Keyword Spotter wrapper.

    Uses sherpa-onnx's pre-trained Zipformer-based KWS models for
    accurate, low-latency keyword detection.

    Requires: pip install sherpa-onnx

    Usage:
        kws = SherpaKWS(
            model_dir="models/kws",
            keywords=["hey_computer", "stop", "go"],
        )
        kws.load()

        # Streaming: feed audio chunks
        kws.reset_stream()
        for chunk in audio_stream:
            result = kws.infer(chunk)
            if result.label != "no_keyword":
                print(f"Detected: {result.label} ({result.confidence:.2f})")
    """

    def __init__(
        self,
        model_dir: str | Path = "models/kws",
        keywords: Optional[list[str]] = None,
        backend: str = "onnx",
        use_gpu: bool = True,
    ):
        """Initialize sherpa-onnx KWS.

        Args:
            model_dir: Directory containing sherpa-onnx model files
                       (encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt).
            keywords: List of keyword strings to detect.
            backend: "onnx" or "tensorrt".
            use_gpu: GPU acceleration.
        """
        super().__init__(model_path=model_dir, backend=backend, use_gpu=use_gpu)
        self._keywords = keywords or ["hey_computer", "stop", "go"]
        self._spotter: Optional[object] = None
        self._stream: Optional[object] = None
        self._has_sherpa = False

    @property
    def sample_rate(self) -> int:
        return KWS_SAMPLE_RATE

    @property
    def task_name(self) -> str:
        return "kws"

    @property
    def keywords(self) -> list[str]:
        return self._keywords

    def reset_stream(self) -> None:
        """Reset the internal streaming state.

        Call when starting a new speech segment.
        """
        if self._spotter is not None and self._has_sherpa:
            self._stream = self._spotter.create_stream()
            logger.debug("KWS stream reset")

    def _load_model(self) -> None:
        """Load the sherpa-onnx KWS model."""
        try:
            import sherpa_onnx
            self._has_sherpa = True
        except ImportError:
            logger.warning(
                "sherpa-onnx not installed. KWS will fall back to mock. "
                "Install with: pip install sherpa-onnx"
            )
            self._has_sherpa = False
            self._is_loaded = True
            return

        model_dir = Path(self._model_path)
        encoder = str(model_dir / "encoder.onnx")
        decoder = str(model_dir / "decoder.onnx")
        joiner = str(model_dir / "joiner.onnx")
        tokens = str(model_dir / "tokens.txt")

        # Check files exist
        for f in [encoder, decoder, joiner, tokens]:
            if not Path(f).exists():
                raise FileNotFoundError(
                    f"KWS model file not found: {f}. "
                    f"Download models first with: "
                    f"python -c \"import sherpa_onnx; sherpa_onnx.download('kws')\""
                )

        # Create keyword spotter config
        config = sherpa_onnx.KeywordSpotterConfig(
            feat_config=sherpa_onnx.FeatureConfig(
                sample_rate=KWS_SAMPLE_RATE,
                feature_dim=80,
            ),
            model_config=sherpa_onnx.OnlineModelConfig(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
            ),
            keywords_file=str(model_dir / "keywords.txt"),
            num_threads=2,
        )

        self._spotter = sherpa_onnx.KeywordSpotter(config)
        self._stream = self._spotter.create_stream()

        logger.info(
            f"sherpa-onnx KWS loaded: {len(self._keywords)} keywords "
            f"({', '.join(self._keywords[:5])}...)"
        )

    def _infer(self, audio: np.ndarray) -> InferenceResult:
        """Run KWS inference on audio chunk.

        Args:
            audio: 1-D float32 array at 16kHz.

        Returns:
            InferenceResult with detected keyword or "no_keyword".
        """
        self.validate_audio(audio)

        # Fallback to mock if sherpa-onnx not available
        if not self._has_sherpa:
            mock = MockKWS(keywords=self._keywords)
            mock._is_loaded = True
            return mock._infer(audio)

        # sherpa-onnx streaming inference
        audio = audio.astype(np.float32)
        self._stream.accept_waveform(KWS_SAMPLE_RATE, audio)

        # Check for results
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)

        # Get the latest result
        result_text = self._spotter.get_result(self._stream).keyword

        if result_text and result_text in self._keywords:
            keyword = result_text
            # sherpa-onnx doesn't provide confidence scores for KWS,
            # so we use a placeholder high confidence
            confidence = 0.95
        else:
            keyword = "no_keyword"
            confidence = 0.0

        return InferenceResult(
            task=self.task_name,
            label=keyword,
            confidence=float(confidence),
            metadata={"keywords": self._keywords, "backend": "sherpa-onnx"},
        )
