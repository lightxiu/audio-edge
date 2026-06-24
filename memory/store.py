"""LanceDB vector store for event memory.

LanceDB is chosen over ChromaDB for its disk-based storage (low RAM)
and embedded deployment model (no server process needed).

If LanceDB is not installed, falls back to an in-memory SQLite store.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

DB_PATH = Path("data/event_memory.lance")


class EventStore:
    """LanceDB-backed event store with vector search.

    Parameters
    ----------
    db_path : str or Path
        Path to the LanceDB database directory.
    embedding_dim : int
        Dimension of embedding vectors.
    """

    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        embedding_dim: int = 384,
    ) -> None:
        self._db_path = Path(db_path)
        self._dim = embedding_dim
        self._db: object | None = None
        self._table: object | None = None
        self._fallback: list[dict] = []  # In-memory fallback
        self._use_fallback = False

    # -- Public API -------------------------------------------------------
    def open(self) -> None:
        """Open or create the database."""
        try:
            import lancedb
        except ImportError:
            self._use_fallback = True
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_path))

        # Create or open events table
        if "events" not in self._db.table_names():
            import pyarrow as pa

            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("timestamp", pa.float64()),
                pa.field("module", pa.string()),       # "vision" | "audio" | "system"
                pa.field("task", pa.string()),         # e.g. "kws", "yolo", "cpu_alert"
                pa.field("label", pa.string()),        # e.g. "person", "speech", "85°C"
                pa.field("text", pa.string()),         # Human-readable description
                pa.field("vector", pa.list_(pa.float32(), self._dim)),
                pa.field("metadata", pa.string()),     # JSON blob for extra data
            ])
            self._table = self._db.create_table("events", schema=schema)
        else:
            self._table = self._db.open_table("events")

    def insert(
        self,
        text: str,
        vector: list[float],
        module: str = "",
        task: str = "",
        label: str = "",
        metadata: dict | None = None,
    ) -> str:
        """Insert one event. Returns the event ID."""
        import json
        import uuid

        event_id = uuid.uuid4().hex[:12]

        row = {
            "id": event_id,
            "timestamp": time.time(),
            "module": module,
            "task": task,
            "label": label,
            "text": text,
            "vector": vector,
            "metadata": json.dumps(metadata or {}, ensure_ascii=False),
        }

        if self._use_fallback:
            self._fallback.append(row)
            return event_id

        self._table.add([row])
        return event_id

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic search — return top_k most similar events."""
        if self._use_fallback:
            return self._fallback_search(query_vector, top_k)

        results = (
            self._table.search(query_vector)
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        return results

    def query_by_time(
        self,
        start: float,
        end: float | None = None,
        module: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query events within a time range, optionally filtered by module."""
        if end is None:
            end = time.time()

        if self._use_fallback:
            results = [
                r for r in self._fallback
                if start <= r["timestamp"] <= end
                and (module is None or r["module"] == module)
            ]
            return sorted(results, key=lambda r: r["timestamp"], reverse=True)[:limit]

        import lancedb

        query = (
            self._table.search()
            .where(f"timestamp >= {start} AND timestamp <= {end}")
        )
        if module:
            query = query.where(f"module = '{module}'")
        return query.limit(limit).to_list()

    def count(self) -> int:
        """Return total event count."""
        if self._use_fallback:
            return len(self._fallback)
        return self._table.count_rows()

    def close(self) -> None:
        """Close the database (LanceDB auto-closes, but explicit is cleaner)."""
        self._db = None
        self._table = None

    # -- Helpers ----------------------------------------------------------
    def _fallback_search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Brute-force cosine search over in-memory events."""
        import numpy as np

        if not self._fallback:
            return []

        qv = np.array(query_vector)
        scored = []
        for r in self._fallback:
            rv = np.array(r["vector"])
            sim = np.dot(qv, rv) / (np.linalg.norm(qv) * np.linalg.norm(rv) + 1e-8)
            scored.append((float(sim), r))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, r in scored[:top_k]:
            r_copy = dict(r)
            r_copy["_distance"] = 1.0 - sim
            results.append(r_copy)
        return results
