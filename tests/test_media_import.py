import numpy as np
import pytest
from fastapi.testclient import TestClient

from dashboard.server import app, configure_review_store
from src.models.base import InferenceResult
from src.output.review import ReviewStore
from src.pipeline import media_import
from src.utils.config import load_config


def test_timeline_offset_and_timezone():
    samples = np.ones(16000, dtype=np.float32)
    aligned = media_import.align_audio(samples, 10.25, 10.0)
    assert len(aligned) == 20000
    assert np.all(aligned[:4000] == 0)
    assert np.all(aligned[4000:] == 1)
    assert len(media_import.align_audio(samples, 9.75, 10.0)) == 12000
    assert media_import.recording_epoch("2026-10-29T14:00:00+08:00") == media_import.recording_epoch(
        "2026-10-29T06:00:00Z"
    )
    with pytest.raises(ValueError, match="timezone"):
        media_import.recording_epoch("2026-10-29T14:00:00")


def test_same_file_audio_and_video_evidence(tmp_path, monkeypatch):
    media = tmp_path / "recording.mp4"
    media.touch()
    config = load_config("configs/windows_sound_events.yaml")
    config["review"]["db_path"] = str(tmp_path / "review.sqlite3")
    config["review"]["evidence_dir"] = str(tmp_path / "clips")
    config["sed"]["model_path"] = str(media)  # load/infer are mocked; only existence is checked
    config["sed"]["cooldown_sec"] = 10
    monkeypatch.setattr(media_import, "probe_media", lambda path: {
        "format": {"duration": "4.0", "start_time": "10.0"},
        "streams": [
            {"codec_type": "video", "index": 0, "start_time": "10.0"},
            {"codec_type": "audio", "index": 1, "start_time": "10.5"},
        ],
    })
    monkeypatch.setattr(media_import, "decode_audio", lambda *args: np.zeros(56000, dtype=np.float32))
    calls = []

    def fake_clip(path, index, center, duration, dest):
        calls.append((index, center, duration))
        dest.write_bytes(b"video")

    monkeypatch.setattr(media_import, "extract_video", fake_clip)
    monkeypatch.setattr(media_import.YAMNetSED, "load", lambda self: None)
    monkeypatch.setattr(media_import.YAMNetSED, "unload", lambda self: None)
    labels = iter(["Silence", "Silence", "Knock", "Knock", "Knock", "Silence", "Silence"])
    monkeypatch.setattr(
        media_import.YAMNetSED, "infer",
        lambda self, audio: InferenceResult(task="sed", label=next(labels), confidence=0.75),
    )

    outcome = media_import.import_media(media, "2026-10-29T14:00:00+08:00", config, "camera-1")
    assert outcome["has_video"] and len(outcome["candidates"]) == 1
    candidate = outcome["candidates"][0]
    assert candidate["offset_sec"] == 2.0  # 0.5s stream offset + second confirmed window center
    assert calls == [(0, 2.0, 4.0)]
    store = ReviewStore(config["review"]["db_path"], config["review"]["evidence_dir"])
    assert store.get(candidate["id"])["timestamp"] == pytest.approx(
        media_import.recording_epoch("2026-10-29T14:00:00+08:00") + 2.0
    )
    configure_review_store(store)
    try:
        client = TestClient(app)
        association = client.get("/api/associations").json()[0]
        assert association["relationship"] == "same_media_file"
        assert association["media_evidence"]["source_id"] == "camera-1"
        assert client.get(f"/api/candidates/{candidate['id']}/video").content == b"video"
        assert client.get("/api/candidates/missing/video").status_code == 404
    finally:
        configure_review_store(None)


def test_audio_only_and_missing_track(tmp_path, monkeypatch):
    media = tmp_path / "recording.wav"
    media.touch()
    config = load_config("configs/windows_sound_events.yaml")
    config["review"]["db_path"] = str(tmp_path / "review.sqlite3")
    config["review"]["evidence_dir"] = str(tmp_path / "clips")
    config["sed"]["model_path"] = str(media)
    monkeypatch.setattr(media_import, "probe_media", lambda path: {
        "format": {"duration": "1.5", "start_time": "0"}, "streams": [{"codec_type": "audio", "index": 0}],
    })
    monkeypatch.setattr(media_import, "decode_audio", lambda *args: np.zeros(24000, dtype=np.float32))
    monkeypatch.setattr(media_import.YAMNetSED, "load", lambda self: None)
    monkeypatch.setattr(media_import.YAMNetSED, "unload", lambda self: None)
    monkeypatch.setattr(media_import.YAMNetSED, "infer", lambda self, audio: InferenceResult(
        task="sed", label="Silence", confidence=1.0,
    ))
    outcome = media_import.import_media(media, "2026-10-29T06:00:00Z", config)
    assert not outcome["has_video"] and not outcome["candidates"]
    monkeypatch.setattr(media_import, "probe_media", lambda path: {
        "format": {"duration": "1.5"}, "streams": [{"codec_type": "video", "index": 0}],
    })
    with pytest.raises(ValueError, match="No audio stream"):
        media_import.import_media(media, "2026-10-29T06:00:00Z", config)


def test_ffprobe_validation(tmp_path, monkeypatch):
    class Response:
        stdout = b'{"format":{"duration":"601"},"streams":[]}'

    monkeypatch.setattr(media_import, "_binary", lambda name: name)
    monkeypatch.setattr(media_import.subprocess, "run", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="duration"):
        media_import.probe_media(tmp_path / "long.mp4")
