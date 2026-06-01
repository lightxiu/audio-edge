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
    """Collects inference latency measurements and computes percentile statistics."""

    measurements: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, task: str, latency_sec: float) -> None:
        self.measurements[task].append(latency_sec * 1000)

    @contextmanager
    def measure(self, task: str):
        t0 = time.perf_counter()
        yield
        self.record(task, time.perf_counter() - t0)

    def stats(self, task: str) -> LatencyStats:
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
