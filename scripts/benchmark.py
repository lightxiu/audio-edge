#!/usr/bin/env python3
"""End-to-end latency benchmark for audio-edge models.

Measures inference latency (p50, p95, p99) for each model on the
current platform. Run on Jetson Orin NX for production numbers.

Usage:
    python scripts/benchmark.py                # Benchmark all loaded models
    python scripts/benchmark.py --model vad    # Benchmark VAD only
    python scripts/benchmark.py --runs 1000    # 1000 inference runs
    python scripts/benchmark.py --output bench.json  # Save to JSON
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import MetricsCollector

logger = get_logger(__name__)


def benchmark_vad(runs: int = 500, model_path: str = "models/vad/silero_vad.onnx") -> dict:
    """Benchmark Silero VAD latency."""
    from src.models.vad import SileroVAD

    if not Path(model_path).exists():
        return {"error": f"Model not found: {model_path}"}

    vad = SileroVAD(model_path)
    vad.load()

    collector = MetricsCollector()
    audio = np.random.randn(512).astype(np.float32)  # 32ms chunk

    # Warmup
    for _ in range(50):
        vad.infer(audio)

    # Benchmark
    for _ in range(runs):
        vad.reset_state()  # Fresh state each run
        with collector.measure("vad"):
            vad.infer(audio)

    vad.unload()
    return _stats_to_dict(collector.stats("vad"))


def benchmark_kws(runs: int = 200, **kwargs) -> dict:
    """Benchmark KWS (mock or sherpa-onnx) latency."""
    from src.models.kws import MockKWS

    kws = MockKWS()
    kws.load()

    collector = MetricsCollector()
    audio = np.random.randn(512).astype(np.float32)

    for _ in range(20):
        kws.infer(audio)

    for _ in range(runs):
        with collector.measure("kws"):
            kws.infer(audio)

    kws.unload()
    return _stats_to_dict(collector.stats("kws"))


def benchmark_sed(runs: int = 200, **kwargs) -> dict:
    """Benchmark SED latency (mock by default)."""
    from src.models.sed import MockSED

    sed = MockSED()
    sed.load()

    collector = MetricsCollector()
    audio = np.random.randn(16000).astype(np.float32)  # 1 second

    for _ in range(20):
        sed.infer(audio)

    for _ in range(runs):
        with collector.measure("sed"):
            sed.infer(audio)

    sed.unload()
    return _stats_to_dict(collector.stats("sed"))


def benchmark_asc(runs: int = 200, **kwargs) -> dict:
    """Benchmark ASC latency (mock by default)."""
    from src.models.asc import MockASC

    asc = MockASC()
    asc.load()

    collector = MetricsCollector()
    audio = np.random.randn(48000).astype(np.float32)  # 3 seconds

    for _ in range(20):
        asc.infer(audio)

    for _ in range(runs):
        with collector.measure("asc"):
            asc.infer(audio)

    asc.unload()
    return _stats_to_dict(collector.stats("asc"))


BENCHMARKS = {
    "vad": benchmark_vad,
    "kws": benchmark_kws,
    "sed": benchmark_sed,
    "asc": benchmark_asc,
}


def _stats_to_dict(stats) -> dict:
    """Convert LatencyStats to a serializable dict."""
    return {
        "count": stats.count,
        "min_ms": round(stats.min_ms, 3),
        "max_ms": round(stats.max_ms, 3),
        "mean_ms": round(stats.mean_ms, 3),
        "p50_ms": round(stats.p50_ms, 3),
        "p95_ms": round(stats.p95_ms, 3),
        "p99_ms": round(stats.p99_ms, 3),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark audio-edge model latency")
    parser.add_argument("--model", type=str, default=None, help="Benchmark specific model")
    parser.add_argument("--runs", type=int, default=500, help="Number of inference runs")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    setup_logging(level="INFO")

    logger.info("=" * 50)
    logger.info("audio-edge Benchmark")
    logger.info(f"  Runs per model: {args.runs}")
    logger.info("=" * 50)

    results = {}

    if args.model:
        tasks = [(args.model, BENCHMARKS[args.model])]
    else:
        tasks = BENCHMARKS.items()

    for name, benchmark_fn in tasks:
        logger.info(f"\nBenchmarking {name.upper()}...")
        try:
            stats = benchmark_fn(runs=args.runs)
            results[name] = stats

            if "error" in stats:
                logger.warning(f"  {name}: {stats['error']}")
            else:
                logger.info(
                    f"  {name}: mean={stats['mean_ms']:.2f}ms, "
                    f"p50={stats['p50_ms']:.2f}ms, "
                    f"p95={stats['p95_ms']:.2f}ms, "
                    f"p99={stats['p99_ms']:.2f}ms"
                )
        except Exception as e:
            logger.error(f"  {name} benchmark failed: {e}")
            results[name] = {"error": str(e)}

    # Summary table
    logger.info(f"\n{'=' * 50}")
    logger.info("Summary")
    logger.info(f"{'=' * 50}")
    for name, stats in results.items():
        if "error" not in stats:
            logger.info(
                f"  {name:<6} p50={stats['p50_ms']:>8.2f}ms  "
                f"p95={stats['p95_ms']:>8.2f}ms  "
                f"p99={stats['p99_ms']:>8.2f}ms"
            )

    # Save to file
    if args.output:
        output = {
            "platform": sys.platform,
            "runs": args.runs,
            "timestamp": time.time(),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
