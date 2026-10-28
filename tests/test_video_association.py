import time

import numpy as np
from fastapi.testclient import TestClient

from dashboard.server import app, configure_review_store
from src.models.base import InferenceResult
from src.output.review import ReviewStore


def video_payload(timestamp_ms):
    return {
        "type": "abnormality",
        "timestamp": timestamp_ms,
        "count": 1,
        "objects": [{"label": 2, "name": "fish", "probability": 0.8, "box": [1, 2, 30, 40]}],
        "message": "abnormality: 1 object(s) detected",
    }


def test_cpp_video_event_ingest_and_temporal_association(tmp_path):
    store = ReviewStore(tmp_path / "events.sqlite3", tmp_path / "evidence")
    now = time.time()
    audio_id = store.add(
        InferenceResult(task="sed", label="Knock", confidence=0.7, timestamp=now),
        np.zeros(16000, dtype=np.float32), 16000,
    )
    configure_review_store(store)
    try:
        client = TestClient(app)
        nearby = video_payload(round((now + 2) * 1000))
        far = video_payload(round((now + 20) * 1000))
        response = client.post("/api/video-events?source_id=camera-1", json=nearby)
        assert response.status_code == 201
        assert response.json()["source_id"] == "camera-1"
        assert client.post("/api/video-events?source_id=camera-1", json=nearby).json()["id"] == response.json()["id"]
        assert client.post("/api/video-events?source_id=camera-1", json=far).status_code == 201
        assert len(client.get("/api/video-events").json()) == 2
        linked = client.get("/api/associations?window_sec=5").json()[0]
        assert linked["audio"]["id"] == audio_id
        assert linked["relationship"] == "temporal_only"
        assert [event["id"] for event in linked["nearby_video"]] == [response.json()["id"]]
        assert client.get("/api/associations?window_sec=1").json()[0]["nearby_video"] == []
        assert len(ReviewStore(store.db_path, store.evidence_dir).list_video_events()) == 2
    finally:
        configure_review_store(None)


def test_video_only_and_invalid_payload(tmp_path):
    store = ReviewStore(tmp_path / "events.sqlite3", tmp_path / "evidence")
    configure_review_store(store)
    try:
        client = TestClient(app)
        payload = video_payload(1760000000000)
        assert client.post("/api/video-events", json=payload).status_code == 201
        assert client.get("/api/associations").json() == []
        assert len(client.get("/api/video-events").json()) == 1
        assert client.post("/api/video-events", json={**payload, "timestamp": 1760000000}).status_code == 422
        assert client.post("/api/video-events", json={**payload, "count": 2}).status_code == 422
        assert client.get("/api/associations?window_sec=61").status_code == 422
    finally:
        configure_review_store(None)
