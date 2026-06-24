"""Event indexer — converts pipeline events to text embeddings and stores them.

This is the bridge between the audio/video/system pipelines and the
RAG memory store.  Every event that flows through the system can be
indexed for later semantic retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memory.embed import Embedder
from memory.store import EventStore
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Singleton instances
_embedder: Embedder | None = None
_store: EventStore | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
        _embedder.load()
    return _embedder


def _get_store() -> EventStore:
    global _store
    if _store is None:
        _store = EventStore()
        _store.open()
    return _store


def index_event(event: dict[str, Any]) -> str | None:
    """Index a pipeline event into the vector store.

    Parameters
    ----------
    event : dict
        An event dict with at least: ``module``, ``label``, ``summary``.
        Example::

            {
                "module": "vision",
                "label": "person",
                "summary": {"person": 2, "car": 1},
                "timestamp": 1719234567.89,
            }

    Returns
    -------
    str or None
        Event ID if indexed successfully, None otherwise.
    """
    text = _event_to_text(event)
    if not text.strip():
        return None

    try:
        embedder = _get_embedder()
        store = _get_store()

        vector = embedder.embed(text).tolist()

        event_id = store.insert(
            text=text,
            vector=vector,
            module=event.get("module", ""),
            task=event.get("task", ""),
            label=event.get("label", ""),
            metadata=event,
        )
        return event_id
    except Exception as e:
        logger.error(f"Failed to index event: {e}")
        return None


def _event_to_text(event: dict) -> str:
    """Convert an event dict to a human-readable description for embedding."""
    module = event.get("module", "")
    label = event.get("label", "")
    task = event.get("task", "")
    summary = event.get("summary", {})

    parts = []

    if module == "vision":
        if summary:
            items = [f"{cnt} {cls}" for cls, cnt in summary.items()]
            parts.append(f"摄像头检测到: {', '.join(items)}")
        else:
            parts.append(f"摄像头事件: {label}")

    elif module in ("audio", "vad", "kws", "sed", "asc"):
        task_name = {"vad": "语音活动", "kws": "关键词", "sed": "声音事件", "asc": "场景分类"}.get(task, task)
        parts.append(f"{task_name}: {label}")

    elif module == "system":
        parts.append(f"系统状态: {label}")

    else:
        parts.append(f"{module}: {label}")

    return "。".join(parts)
