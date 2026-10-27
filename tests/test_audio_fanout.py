import time

from src.models.base import InferenceResult
from src.pipeline.orchestrator import Orchestrator
from src.utils.config import load_config


def test_vad_and_sed_receive_same_audio_without_competing_reads(monkeypatch):
    config = load_config("configs/windows_sound_events.yaml")
    config["_mock_audio"] = True
    config["output"]["console"] = False
    config["output"]["jsonl_path"] = None
    config["output"]["dashboard"] = False
    config["review"]["enabled"] = False
    config["logging"]["file"] = None
    config["report"]["enabled"] = False
    config["sed"]["interval_sec"] = 0
    orchestrator = Orchestrator(config)
    received = {"vad": 0, "sed": 0}

    def vad_infer(chunk):
        received["vad"] += len(chunk)
        return InferenceResult(task="vad", label="silence", confidence=0.0)

    def sed_infer(chunk):
        received["sed"] += len(chunk)
        return InferenceResult(task="sed", label="Silence", confidence=1.0)

    monkeypatch.setattr(orchestrator._vad, "load", lambda: None)
    monkeypatch.setattr(orchestrator._vad, "unload", lambda: None)
    monkeypatch.setattr(orchestrator._vad, "infer", vad_infer)
    monkeypatch.setattr(orchestrator._sed, "load", lambda: None)
    monkeypatch.setattr(orchestrator._sed, "unload", lambda: None)
    monkeypatch.setattr(orchestrator._sed, "infer", sed_infer)

    orchestrator.start()
    try:
        deadline = time.monotonic() + 3
        while received["sed"] == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        orchestrator.stop()

    assert received["vad"] >= 16000
    assert received["sed"] == 16000
