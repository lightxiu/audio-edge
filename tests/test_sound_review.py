import wave

import numpy as np
from fastapi.testclient import TestClient

from dashboard.server import app, configure_review_store
from src.models.base import InferenceResult
from src.models.sed import YAMNetSED
from src.output.review import ReviewStore
from src.pipeline.sed_confirmation import SEDConfirmation


def test_non_target_winner_does_not_force_target():
    class Input:
        name = "audio"
        shape = [None]

    class Model:
        def get_inputs(self):
            return [Input()]

        def run(self, names, inputs):
            return np.array([[0.8, 0.3]], dtype=np.float32), None

    detector = YAMNetSED(threshold=0.2, target_classes=["Knock"])
    detector._model = Model()
    detector._labels = {0: "Speech", 1: "Knock"}
    detector._target_indices = [1]
    assert detector._infer(np.zeros(16000, dtype=np.float32)).label == "Silence"
    detector._model.run = lambda names, inputs: (np.array([[0.1, 0.7]], dtype=np.float32), None)
    assert detector._infer(np.zeros(16000, dtype=np.float32)).label == "Knock"


def test_confirmation_resets_after_other_sound():
    gate = SEDConfirmation(2)
    def result(label):
        return InferenceResult(task="sed", label=label, confidence=0.7)
    assert not gate.update(result("Knock"))
    assert not gate.update(result("Silence"))
    assert not gate.update(result("Knock"))
    assert gate.update(result("Knock"))
    assert not gate.update(result("Doorbell"))


def test_evidence_and_review_api(tmp_path):
    store = ReviewStore(tmp_path / "reviews" / "events.db", tmp_path / "evidence")
    result = InferenceResult(task="sed", label="Knock", confidence=0.75)
    candidate_id = store.add(result, np.zeros(16000, dtype=np.float32), 16000)
    with wave.open(str(store.evidence_dir / f"{candidate_id}.wav"), "rb") as wav:
        assert wav.getnframes() == 16000
        assert wav.getframerate() == 16000

    configure_review_store(store)
    try:
        client = TestClient(app)
        assert client.get("/api/candidates?status=pending").json()[0]["id"] == candidate_id
        assert client.get(f"/api/candidates/{candidate_id}/audio").content[:4] == b"RIFF"
        assert client.get("/api/candidates/../../pyproject.toml/audio").status_code != 200
        assert client.patch(f"/api/candidates/{candidate_id}", json={"status": "pending"}).status_code == 422
        reviewed = client.patch(f"/api/candidates/{candidate_id}", json={"status": "confirmed"})
        assert reviewed.json()["status"] == "confirmed"
        assert not client.get("/api/candidates?status=pending").json()
        assert ReviewStore(store.db_path, store.evidence_dir).get(candidate_id)["status"] == "confirmed"
    finally:
        configure_review_store(None)
