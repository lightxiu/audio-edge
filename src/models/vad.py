"""Silero Voice Activity Detection (VAD) model wrapper.

Silero VAD v5 is a stateful RNN-based model that outputs speech probability
per audio frame. It runs in real-time with <5ms latency on CPU.

Model source: https://github.com/snakers4/silero-vad
License: MIT
"""

from collections.abc import Generator
from pathlib import Path

import numpy as np

from src.models.base import BaseModel, InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Silero VAD model constants
SILERO_SAMPLE_RATE = 16000
SILERO_WINDOW_SIZE = 512  # 32ms @ 16kHz — supports 256, 512, 768 samples


class SileroVAD(BaseModel):
    """Silero VAD v5 ONNX model wrapper.

    This is a stateful VAD: the RNN hidden state is maintained across calls
    to provide accurate speech/non-speech decisions on streaming audio.

    Usage:
        vad = SileroVAD("models/vad/silero_vad.onnx")
        vad.load()

        # Streaming: feed 512-sample chunks
        for chunk in audio_stream:
            result = vad.infer(chunk)
            if result.confidence > 0.9:
                print("Speech detected")

        # Or process a full utterance
        segments = vad.detect_segments(long_audio, min_speech_ms=250)
    """

    def __init__(
        self,
        model_path: str | Path = "models/vad/silero_vad.onnx",
        backend: str = "onnx",
        use_gpu: bool = False,  # VAD is so lightweight, CPU is fine
        threshold: float = 0.5,
    ):
        """Initialize Silero VAD.

        Args:
            model_path: Path to silero_vad.onnx.
            backend: "onnx" or "tensorrt".
            use_gpu: GPU acceleration (usually unnecessary for VAD).
            threshold: Speech probability threshold (0.0–1.0).
        """
        super().__init__(model_path=model_path, backend=backend, use_gpu=use_gpu)
        self._threshold = threshold
        self._state: np.ndarray | None = None
        self._sample_rate_tensor: np.ndarray | None = None

    @property
    def sample_rate(self) -> int:
        return SILERO_SAMPLE_RATE

    @property
    def task_name(self) -> str:
        return "vad"

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    def reset_state(self) -> None:
        """Reset the VAD's RNN hidden state.

        Call this when starting a new audio stream or after a pause.
        """
        if self._init_state is not None:
            self._state = self._init_state.copy()
        logger.debug("VAD state reset")

    def _load_model(self) -> None:
        """Load the Silero VAD ONNX model."""
        import onnxruntime as ort

        providers = self._get_providers()
        self._model = ort.InferenceSession(
            str(self._model_path),
            providers=providers,
        )

        # Initialize the sample rate tensor (constant scalar int64)
        self._sample_rate_tensor = np.array(SILERO_SAMPLE_RATE, dtype=np.int64)

        # Store input/output names by ACTUAL name (not position)
        # Silero VAD ONNX inputs: input(float), state(float), sr(int64)
        input_names = [inp.name for inp in self._model.get_inputs()]

        self._input_name = "input"
        self._state_name = "state"
        self._sr_name = "sr"
        self._output_name = "output"
        self._state_out_name = "stateN"

        # Verify all expected names exist
        for name in [self._input_name, self._state_name, self._sr_name]:
            if name not in input_names:
                raise RuntimeError(f"Expected input '{name}' not found in model. Available inputs: {input_names}")

        # Create initial RNN state: shape [2, 1, 128]
        # The model uses a 2-layer GRU with 128 hidden units.
        # ONNX reports dynamic dims as None/strings — map to concrete values.
        raw_shape = self._model.get_inputs()[1].shape  # e.g. [2, None, 128]
        state_shape = tuple(1 if (d is None or isinstance(d, str)) else d for d in raw_shape)
        self._init_state = np.zeros(state_shape, dtype=np.float32)
        self._state = self._init_state.copy()

        logger.debug(f"Silero VAD loaded: inputs={input_names}, state_shape={state_shape}, providers={providers}")

    def _infer(self, audio: np.ndarray) -> InferenceResult:
        """Run VAD inference on a single audio chunk.

        Args:
            audio: 1-D float32 array of audio samples (any length supported).

        Returns:
            InferenceResult with label="speech" or "silence".
        """
        self.validate_audio(audio)

        # Reshape to [batch=1, samples]
        if audio.ndim == 1:
            audio_input = audio.reshape(1, -1)
        else:
            audio_input = audio

        # Ensure float32
        if audio_input.dtype != np.float32:
            audio_input = audio_input.astype(np.float32)

        # Run ONNX inference
        ort_inputs = {
            self._input_name: audio_input,
            self._sr_name: self._sample_rate_tensor,
            self._state_name: self._state,
        }
        ort_outputs = self._model.run(
            [self._output_name, self._state_out_name],
            ort_inputs,
        )

        probability = float(ort_outputs[0].squeeze())
        self._state = ort_outputs[1]  # Update hidden state

        is_speech = probability >= self._threshold
        label = "speech" if is_speech else "silence"

        return InferenceResult(
            task=self.task_name,
            label=label,
            confidence=probability,
            metadata={
                "is_speech": is_speech,
                "samples": len(audio),
            },
        )

    def detect_segments(
        self,
        audio: np.ndarray,
        min_speech_duration_ms: float = 250,
        min_silence_duration_ms: float = 300,
        speech_pad_ms: float = 200,
    ) -> Generator[dict[str, float], None, None]:
        """Detect speech segments in a longer audio recording.

        This is a non-streaming convenience method that processes the entire
        audio and yields speech segment boundaries.

        Args:
            audio: 1-D float32 array at 16kHz.
            min_speech_duration_ms: Minimum speech segment length.
            min_silence_duration_ms: Minimum silence to end a segment.
            speech_pad_ms: Padding added to segment start/end.

        Yields:
            Dict with keys: start_sec, end_sec, duration_sec, confidence.
        """
        self.reset_state()

        window = SILERO_WINDOW_SIZE
        stride = window  # Non-overlapping for segment detection

        # Convert thresholds to frames
        min_speech_frames = int(min_speech_duration_ms / 1000 * self.sample_rate / stride)
        min_silence_frames = int(min_silence_duration_ms / 1000 * self.sample_rate / stride)
        pad_samples = int(speech_pad_ms / 1000 * self.sample_rate)

        # Scan audio
        probs = []
        for i in range(0, len(audio), stride):
            chunk = audio[i : i + window]
            if len(chunk) < window:
                chunk = np.pad(chunk, (0, window - len(chunk)), mode="constant")

            result = self.infer(chunk)
            probs.append(result.confidence)

        # Find speech segments using hysteresis
        speech_mask = np.array([p >= self._threshold for p in probs], dtype=bool)
        segments = self._hysteresis_segmentation(speech_mask, min_speech_frames, min_silence_frames)

        for start_frame, end_frame in segments:
            start_sample = max(0, start_frame * stride - pad_samples)
            end_sample = min(len(audio), end_frame * stride + pad_samples)
            segment_probs = probs[start_frame:end_frame]

            yield {
                "start_sec": start_sample / self.sample_rate,
                "end_sec": end_sample / self.sample_rate,
                "duration_sec": (end_sample - start_sample) / self.sample_rate,
                "confidence": float(np.mean(segment_probs)) if segment_probs else 0.0,
            }

    @staticmethod
    def _hysteresis_segmentation(
        mask: np.ndarray,
        min_speech_frames: int,
        min_silence_frames: int,
    ) -> list[tuple[int, int]]:
        """Segment a binary mask using hysteresis thresholds.

        Args:
            mask: Boolean array (True = speech).
            min_speech_frames: Minimum consecutive True frames for a segment.
            min_silence_frames: Minimum consecutive False frames to split.

        Returns:
            List of (start_frame, end_frame) tuples (end is exclusive).
        """
        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0
        speech_count = 0

        for i, is_speech in enumerate(mask):
            if is_speech:
                speech_count += 1
                silence_count = 0
                if not in_speech and speech_count >= min_speech_frames:
                    in_speech = True
                    speech_start = i - speech_count + 1
            else:
                speech_count = 0
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        segments.append((speech_start, i - silence_count + 1))
                        in_speech = False
                        silence_count = 0

        if in_speech:
            segments.append((speech_start, len(mask)))

        return segments

    def _get_providers(self) -> list[str]:
        """Get ONNX Runtime execution providers based on backend setting."""
        import onnxruntime as ort

        available = ort.get_available_providers()

        if self._backend == "tensorrt" and "TensorrtExecutionProvider" in available:
            return ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        elif self._use_gpu and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
