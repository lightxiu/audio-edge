"""STT (Speech-to-Text) via sherpa-onnx Whisper.

Thin wrapper that reuses the project's existing sherpa-onnx integration
for streaming speech recognition.

For the edge sentinel, we use a non-streaming (utterance-based) approach:
wait for VAD to detect end-of-speech, then transcribe the accumulated audio.
"""

from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SpeechRecognizer:
    """Speech recognition via sherpa-onnx Whisper.

    Parameters
    ----------
    model_dir : str
        Path to sherpa-onnx Whisper model directory.
    """

    def __init__(self, model_dir: str = "models/stt/whisper-tiny") -> None:
        self._model_dir = model_dir
        self._recognizer: object | None = None  # sherpa_onnx.OnlineRecognizer
        self._is_loaded = False
        self._use_mock = False

    def load(self) -> None:
        """Load the Whisper model."""
        if self._is_loaded:
            return

        try:
            import sherpa_onnx

            config = sherpa_onnx.OnlineRecognizerConfig(
                model_config=sherpa_onnx.OnlineModelConfig(
                    whisper=sherpa_onnx.WhisperModelConfig(
                        encoder=f"{self._model_dir}/encoder.int8.onnx",
                        decoder=f"{self._model_dir}/decoder.int8.onnx",
                        language="zh",
                    ),
                ),
            )
            self._recognizer = sherpa_onnx.OnlineRecognizer(config)
            logger.info("STT Whisper model loaded")
        except Exception as e:
            logger.warning(f"STT model not available ({e}), using mock")
            self._use_mock = True

        self._is_loaded = True

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe an audio chunk (non-streaming).

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 audio samples at 16kHz.
        sample_rate : int
            Audio sample rate.

        Returns
        -------
        str
            Transcribed text, or empty string if nothing detected.
        """
        if self._use_mock or self._recognizer is None:
            return self._mock_transcribe(audio)

        # sherpa-onnx online recognition
        self._recognizer.accept_waveform(sample_rate, audio)
        self._recognizer.input_finished()

        text = self._recognizer.text.strip()
        return text

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @staticmethod
    def _mock_transcribe(audio: np.ndarray) -> str:
        """Mock STT — returns empty (no speech detected)."""
        # Check if audio is non-silent
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 0.01:
            return ""
        return "[mock STT: 语音输入]"
