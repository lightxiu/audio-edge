import time

import numpy as np
from fastapi.testclient import TestClient

from agent.inspection import InspectionAgent
from dashboard.server import app, configure_review_store
from src.models.base import InferenceResult
from src.output.review import ReviewStore


def video_event(timestamp):
    return {"type": "abnormality", "timestamp": round(timestamp * 1000), "count": 1,
            "objects": [{"label": 2, "name": "fish", "probability": 0.8, "box": [1, 2, 30, 40]}],
            "message": "test"}


def test_inspection_evidence_and_relationships(tmp_path, monkeypatch):
    store = ReviewStore(tmp_path / "events.db", tmp_path / "evidence")
    now = time.time() - 30
    audio_id = store.add(InferenceResult(task="sed", label="Knock", confidence=0.8, timestamp=now),
                         np.zeros(16000, dtype=np.float32), 16000)
    video = store.add_video_event(video_event(now + 2), "camera-1")
    monkeypatch.setattr(InspectionAgent, "_rank", lambda self, question, evidence: [audio_id])
    report = InspectionAgent(store).inspect("有什么声音？", now - 10, now + 10)
    assert report["source"] == "ollama"
    assert report["focus_ids"] == [audio_id]
    assert [item["tool"] for item in report["tool_trace"]] == ["candidates_between", "video_events_between"]
    assert report["evidence"][1]["relation"] == "temporal_only"
    assert report["evidence"][1]["nearby_video_ids"] == [video["id"]]
    assert report["evidence"][1]["audio_url"].endswith(f"{audio_id}/audio")

    (store.evidence_dir / "clip.mp4").write_bytes(b"placeholder")
    store.add_media_evidence(audio_id, "recording", 1.5, "clip.mp4")
    report = InspectionAgent(store).inspect("有什么声音？", now - 10, now + 10)
    assert report["evidence"][1]["relation"] == "same_media_file"
    assert report["evidence"][1]["video_url"].endswith(f"{audio_id}/video")


def test_bad_model_ids_fall_back_without_fabricating(tmp_path, monkeypatch):
    store = ReviewStore(tmp_path / "events.db", tmp_path / "evidence")
    now = time.time() - 30
    store.add_video_event(video_event(now), "camera-1")
    monkeypatch.setattr(InspectionAgent, "_rank", lambda self, question, evidence: ["invented"])
    result = InspectionAgent(store).inspect("有视频吗", now - 1, now + 1)
    assert result["source"] == "template"
    assert result["focus_ids"] == [result["evidence"][0]["id"]]
    assert result["summary"].startswith("查询区间内记录音频候选 0 条、视频事件 1 条")
    monkeypatch.setattr(InspectionAgent, "_rank", lambda self, question, evidence: (_ for _ in ()).throw(
        TimeoutError("offline")))
    assert InspectionAgent(store).inspect("有视频吗", now - 1, now + 1)["source"] == "template"
    assert InspectionAgent(store).inspect("没有事件", now - 100, now - 50)["evidence"] == []


def test_inspection_api_bounds_and_no_mutation(tmp_path, monkeypatch):
    store = ReviewStore(tmp_path / "events.db", tmp_path / "evidence")
    now = time.time() - 30
    candidate_id = store.add(InferenceResult(task="sed", label="Alarm", confidence=0.6, timestamp=now),
                             np.zeros(16000, dtype=np.float32), 16000)
    monkeypatch.setattr(InspectionAgent, "_rank", lambda self, question, evidence: [candidate_id])
    configure_review_store(store)
    try:
        client = TestClient(app)
        params = {"question": "有没有警报？", "start": now - 5, "end": now + 5}
        assert client.post("/api/inspection", json=params).json()["focus_ids"] == [candidate_id]
        assert store.get(candidate_id)["status"] == "pending"
        assert client.post("/api/inspection", json={**params, "end": now + 90000}).status_code == 422
        assert client.post("/api/inspection", json={**params, "start": 0}).status_code == 422
        assert client.post("/api/inspection", json={**params, "question": ""}).status_code == 422
    finally:
        configure_review_store(None)
