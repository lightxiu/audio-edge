"""Natural language query over event memory.

Provides a simple semantic search interface: given a user question
in natural language, returns the most relevant historical events.
"""

from __future__ import annotations

import time
from typing import Any

from memory.embed import Embedder
from memory.store import EventStore
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _get_embedder() -> Embedder:
    # Lazily reuse the singleton from indexer
    from memory.indexer import _get_embedder as _ge
    return _ge()


def _get_store() -> EventStore:
    from memory.indexer import _get_store as _gs
    return _gs()


def query_events(
    query: str,
    top_k: int = 5,
    module: str | None = None,
    time_range_hours: float | None = None,
) -> list[dict[str, Any]]:
    """Search events semantically.

    Parameters
    ----------
    query : str
        Natural language query, e.g. ``"有人经过门口吗"``.
    top_k : int
        Number of results to return.
    module : str or None
        Filter by module (``"vision"``, ``"audio"``, ``"system"``).
    time_range_hours : float or None
        Only return events from the last N hours.

    Returns
    -------
    list[dict]
        Matching events sorted by relevance.
    """
    try:
        embedder = _get_embedder()
        store = _get_store()

        qv = embedder.embed(query).tolist()
        results = store.search(qv, top_k=top_k * 2)  # Over-fetch for filtering

        # Filter by module
        if module:
            results = [r for r in results if r.get("module") == module]

        # Filter by time
        if time_range_hours is not None:
            cutoff = time.time() - time_range_hours * 3600
            results = [r for r in results if r.get("timestamp", 0) >= cutoff]

        return results[:top_k]
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []


def query_recent(
    hours: float = 1,
    module: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get recent events without semantic filtering.

    Parameters
    ----------
    hours : float
        Time window in hours.
    module : str or None
        Filter by module.
    limit : int
        Max results.

    Returns
    -------
    list[dict]
        Recent events sorted by timestamp (newest first).
    """
    try:
        store = _get_store()
        cutoff = time.time() - hours * 3600
        return store.query_by_time(start=cutoff, module=module, limit=limit)
    except Exception as e:
        logger.error(f"Query recent failed: {e}")
        return []


def format_query_results(results: list[dict]) -> str:
    """Format query results as a human-readable string for the LLM agent.

    Parameters
    ----------
    results : list[dict]
        Raw query results from ``query_events``.

    Returns
    -------
    str
        Formatted text suitable for passing to an LLM.
    """
    if not results:
        return "没有找到相关事件。"

    lines = []
    for i, r in enumerate(results, 1):
        ts = r.get("timestamp", 0)
        time_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "?"
        text = r.get("text", str(r))
        lines.append(f"{i}. [{time_str}] {text}")

    return "\n".join(lines)
