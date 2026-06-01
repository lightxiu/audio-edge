"""Inference latency tracking and performance metrics.

Collects timing data and computes percentile statistics for
benchmarking model inference performance.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyStats:
    """Latency distribution statistics."""

    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


@dataclass
class MetricsCollector:
    """Collects and analyzes inference latency metrics.

    Usage:
        collector = MetricsCollector()

        # Record manually
        t0 = time.perf_counter()
        model.infer(audio)
        collector.record("vad", time.perf_counter() - t0)

        # Or use context manager
        with collector.measure("kws"):
            model.infer(audio)

        # Get stats
        stats = collector.stats("vad")
        print(f"VAD p95 latency: {stats.p95_ms:.2f}ms")
    """

    measurements: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, task: str, latency_sec: float) -> None:
        """Record a latency measurement.

        Args:
            task: Task name (vad, kws, sed, asc).
            latency_sec: Latency in seconds.
        """
        self.measurements[task].append(latency_sec * 1000)  # Store as ms

    @contextmanager
    def measure(self, task: str):
        """Context manager for measuring inference latency.

        Usage:
            with collector.measure("kws"):
                result = model.infer(audio)
        """
        t0 = time.perf_counter()
        yield
        self.record(task, time.perf_counter() - t0)

    def stats(self, task: str) -> LatencyStats:
        """Compute latency statistics for a task.

        Args:
            task: Task name.

        Returns:
            LatencyStats with percentile breakdown.
        """
        values = np.array(self.measurements.get(task, []))
        if len(values) == 0:
            return LatencyStats()

        return LatencyStats(
            count=len(values),
            min_ms=float(np.min(values)),
            max_ms=float(np.max(values)),
            mean_ms=float(np.mean(values)),
            p50_ms=float(np.percentile(values, 50)),
            p95_ms=float(np.percentile(values, 95)),
            p99_ms=float(np.percentile(values, 99)),
        )

    def summary(self) -> str:
        """Generate a human-readable summary of all tasks.

        Returns:
            Formatted string with per-task statistics.
        """
        lines = []
        lines.append(f"{'Task':<8} {'Count':<8} {'Mean':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'Min':<10} {'Max'}")
        lines.append("-" * 80)

        for task in sorted(self.measurements.keys()):
            s = self.stats(task)
            lines.append(
                f"{task:<8} {s.count:<8} "
                f"{s.mean_ms:<10.2f} {s.p50_ms:<10.2f} {s.p95_ms:<10.2f} "
                f"{s.p99_ms:<10.2f} {s.min_ms:<10.2f} {s.max_ms:.2f}"
            )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all measurements."""
        self.measurements.clear()
