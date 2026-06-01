"""JSON Lines file output backend.

Writes structured event logs in JSONL format for post-hoc analysis.
Each line is a complete JSON object representing one event.
"""

import json
import time
from pathlib import Path
from typing import Optional, TextIO

from src.models.base import InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class JSONLOutput:
    """Writes inference events to a JSONL file."""

    def __init__(self, path: str | Path):
        """Initialize JSONL output.

        Args:
            path: Path to the JSONL file.
        """
        self._path = Path(path)
        self._file: Optional[TextIO] = None
        self._event_count = 0

    def open(self) -> None:
        """Open the output file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        logger.info(f"JSONL output: {self._path}")

    def close(self) -> None:
        """Close the output file."""
        if self._file:
            self._file.close()
            self._file = None
            logger.info(f"JSONL output closed ({self._event_count} events)")

    def emit(self, result: InferenceResult) -> None:
        """Write an inference result as a JSON line.

        Args:
            result: InferenceResult to serialize.
        """
        if not self._file:
            return

        record = {
            "timestamp": result.timestamp,
            "task": result.task,
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "latency_ms": round(result.latency_ms, 2),
        }
        # Include metadata if present
        if result.metadata:
            record["metadata"] = {
                k: v for k, v in result.metadata.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }

        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self._event_count += 1

    @property
    def event_count(self) -> int:
        return self._event_count
