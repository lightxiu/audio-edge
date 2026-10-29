"""Local evidence and durable human review for sound-event candidates."""

from __future__ import annotations

import hashlib
import json
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
            db.execute("""CREATE TABLE IF NOT EXISTS video_events (
                id TEXT PRIMARY KEY, timestamp_ms INTEGER NOT NULL,
                event_type TEXT NOT NULL, source_id TEXT NOT NULL,
                payload TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS video_time_idx ON video_events(timestamp_ms)")
            db.execute("""CREATE TABLE IF NOT EXISTS media_evidence (
                candidate_id TEXT PRIMARY KEY REFERENCES candidates(id),
                source_id TEXT NOT NULL, media_offset_sec REAL NOT NULL,
                video_file TEXT NOT NULL)""")

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

    def candidates_between(self, start: float, end: float, limit: int = 50) -> list[dict]:
        """Read a bounded time window for evidence-backed inspection."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM candidates WHERE timestamp BETWEEN ? AND ? "
                "ORDER BY timestamp DESC, id DESC LIMIT ?", (start, end, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def review(self, candidate_id: str, status: str, note: str = "") -> dict | None:
        if status not in ("confirmed", "rejected"):
            raise ValueError("invalid review status")
        with self._connect() as db:
            db.execute(
                "UPDATE candidates SET status = ?, note = ?, reviewed_at = datetime('now') WHERE id = ?",
                (status, note, candidate_id),
            )
        return self.get(candidate_id)

    def add_video_event(self, event: dict, source_id: str = "unspecified") -> dict:
        """Persist the fish-monitor HTTP payload without claiming synchronized media."""
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        event_id = hashlib.sha256((source_id + "\0" + payload).encode("utf-8")).hexdigest()[:32]
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO video_events (id, timestamp_ms, event_type, source_id, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_id, event["timestamp"], event["type"], source_id, payload),
            )
        return self.get_video_event(event_id)

    def get_video_event(self, event_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM video_events WHERE id = ?", (event_id,)).fetchone()
            return self._video_record(row) if row else None

    @staticmethod
    def _video_record(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "timestamp_ms": row["timestamp_ms"],
            "type": row["event_type"], "source_id": row["source_id"],
            "payload": json.loads(row["payload"]),
        }

    def list_video_events(self, limit: int = 100) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM video_events ORDER BY timestamp_ms DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._video_record(row) for row in rows]

    def video_events_between(self, start: float, end: float, limit: int = 50) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM video_events WHERE timestamp_ms BETWEEN ? AND ? "
                "ORDER BY timestamp_ms DESC, id DESC LIMIT ?",
                (round(start * 1000), round(end * 1000), limit),
            ).fetchall()
            return [self._video_record(row) for row in rows]

    def nearby_video_events(self, timestamp: float, window_sec: float = 5.0) -> list[dict]:
        center_ms = round(timestamp * 1000)
        window_ms = round(window_sec * 1000)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM video_events WHERE timestamp_ms BETWEEN ? AND ? "
                "ORDER BY ABS(timestamp_ms - ?) ASC, id ASC LIMIT 20",
                (center_ms - window_ms, center_ms + window_ms, center_ms),
            ).fetchall()
            return [self._video_record(row) for row in rows]

    def add_media_evidence(self, candidate_id: str, source_id: str, media_offset_sec: float, video_file: str) -> None:
        if not self.get(candidate_id) or Path(video_file).name != video_file:
            raise ValueError("invalid media evidence")
        if not (self.evidence_dir / video_file).is_file():
            raise FileNotFoundError(video_file)
        with self._connect() as db:
            db.execute(
                "INSERT INTO media_evidence (candidate_id, source_id, media_offset_sec, video_file) "
                "VALUES (?, ?, ?, ?)",
                (candidate_id, source_id, media_offset_sec, video_file),
            )

    def get_media_evidence(self, candidate_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT candidate_id, source_id, media_offset_sec, video_file "
                "FROM media_evidence WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            return dict(row) if row else None
