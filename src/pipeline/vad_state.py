"""Confirm VAD transitions only after sustained speech or silence."""

from src.models.base import InferenceResult


class VADState:
    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        release_threshold: float | None = None,
    ) -> None:
        self.threshold = threshold
        self.release_threshold = release_threshold if release_threshold is not None else threshold
        if self.release_threshold > threshold:
            raise ValueError("VAD release threshold must not exceed speech threshold")
        self.speech_samples = max(1, int(sample_rate * min_speech_duration_ms / 1000))
        self.silence_samples = max(1, int(sample_rate * min_silence_duration_ms / 1000))
        self.is_speech = False
        self._pending_samples = 0

    def update(self, result: InferenceResult, samples: int) -> InferenceResult | None:
        is_candidate = (
            result.confidence >= self.release_threshold
            if self.is_speech
            else result.confidence >= self.threshold
        )
        if is_candidate == self.is_speech:
            self._pending_samples = 0
            return None

        self._pending_samples += samples
        minimum = self.silence_samples if self.is_speech else self.speech_samples
        if self._pending_samples < minimum:
            return None

        self.is_speech = is_candidate
        self._pending_samples = 0
        result.label = "speech" if is_candidate else "silence"
        return result
