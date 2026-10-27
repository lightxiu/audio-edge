from src.models.base import InferenceResult
from src.pipeline.vad_state import VADState


def update(state: VADState, probability: float):
    return state.update(InferenceResult(task="vad", label="raw", confidence=probability), 512)


def test_vad_requires_sustained_speech_and_silence():
    state = VADState(16000, threshold=0.5, release_threshold=0.35,
                     min_speech_duration_ms=250, min_silence_duration_ms=300)

    assert all(update(state, 0.8) is None for _ in range(7))
    assert update(state, 0.8).label == "speech"
    assert update(state, 0.3) is None
    assert update(state, 0.4) is None  # Short dip does not end speech.
    assert all(update(state, 0.2) is None for _ in range(9))
    assert update(state, 0.2).label == "silence"


def test_vad_short_speech_burst_resets_candidate():
    state = VADState(16000, 0.5, 250, 300)
    for _ in range(7):
        assert update(state, 0.9) is None
    assert update(state, 0.1) is None
    for _ in range(7):
        assert update(state, 0.9) is None
    assert update(state, 0.9).label == "speech"
