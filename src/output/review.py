"""Local evidence and durable human review for sound-event candidates."""

import sqlite3
import uuid
import wave
from pathlib import Path

import numpy as np

from src.models.base import InferenceResult


class ReviewStore:
    def __init__(self, db_path: str | Path, evidence_dir: str | Path):
        self.db_path = Path(db_path)
        self.evidence_dir = Path(evidence_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, timestamp REAL NOT NULL, label TEXT NOT NULL,
                confidence REAL NOT NULL, evidence TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', reviewed_at TEXT,
                note TEXT NOT NULL DEFAULT '')""")

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def add(self, result: InferenceResult, audio: np.ndarray, sample_rate: int) -> str:
        candidate_id = uuid.uuid4().hex
        evidence = f"{candidate_id}.wav"
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        with wave.open(str(self.evidence_dir / evidence), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        with self._connect() as db:
            db.execute(
                "INSERT INTO candidates (id, timestamp, label, confidence, evidence) VALUES (?, ?, ?, ?, ?)",
                (candidate_id, result.timestamp, result.label, result.confidence, evidence),
            )
        return candidate_id

    def list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        if status is not None and status not in ("pending", "confirmed", "rejected"):
            raise ValueError("invalid status")
        query = "SELECT * FROM candidates"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        with self._connect() as db:
            return [dict(row) for row in db.execute(query + " ORDER BY timestamp DESC LIMIT ?", (*params, limit))]

    def get(self, candidate_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
            return dict(row) if row else None

    def review(self, candidate_id: str, status: str, note: str = "") -> dict | None:
        if status not in ("confirmed", "rejected"):
            raise ValueError("invalid review status")
        with self._connect() as db:
            db.execute(
                "UPDATE candidates SET status = ?, note = ?, reviewed_at = datetime('now') WHERE id = ?",
                (status, note, candidate_id),
            )
        return self.get(candidate_id)
