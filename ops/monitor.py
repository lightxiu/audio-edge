"""System health monitor.

Collects CPU, GPU (Jetson), memory, disk, and temperature metrics.
Designed to run on a polling interval (e.g. every 30s).
"""

from __future__ import annotations

import time
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Thresholds for alerting
THRESHOLDS = {
    "cpu_temp_c": 85,       # °C
    "gpu_temp_c": 85,       # °C
    "memory_percent": 90,   # %
    "disk_percent": 90,     # %
    "cpu_percent": 95,      # %
}


def get_system_status() -> dict[str, Any]:
    """Collect current system metrics.

    Returns
    -------
    dict
        Keys: timestamp, cpu_percent, memory_percent, memory_used_gb,
        memory_total_gb, disk_percent, cpu_temp_c, gpu_temp_c,
        gpu_memory_used_gb, gpu_memory_total_gb, warnings (list[str]).
    """
    status: dict[str, Any] = {
        "timestamp": time.time(),
        "cpu_percent": _get_cpu_percent(),
        "memory_percent": _get_memory_percent(),
        "memory_used_gb": _get_memory_used_gb(),
        "memory_total_gb": _get_memory_total_gb(),
        "disk_percent": _get_disk_percent(),
    }

    # GPU / Jetson-specific
    jetson = _get_jetson_metrics()
    status.update(jetson)

    # Check thresholds
    warnings = []
    for key, threshold in THRESHOLDS.items():
        if key in status and status[key] is not None:
            if status[key] > threshold:
                warnings.append(f"{key}: {status[key]} > {threshold}")
    status["warnings"] = warnings
    status["healthy"] = len(warnings) == 0

    return status


def is_healthy() -> bool:
    """Quick health check — True if no thresholds are exceeded."""
    return get_system_status()["healthy"]


# -- Internal collectors -------------------------------------------------

def _get_cpu_percent() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0.5)
    except Exception:
        return 0.0


def _get_memory_percent() -> float:
    try:
        import psutil
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


def _get_memory_used_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().used / (1024**3), 1)
    except Exception:
        return 0.0


def _get_memory_total_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        return 0.0


def _get_disk_percent() -> float:
    try:
        import psutil
        return psutil.disk_usage("/").percent
    except Exception:
        return 0.0


def _get_jetson_metrics() -> dict[str, Any]:
    """Get Jetson-specific GPU metrics via tegrastats or jtop."""
    metrics: dict[str, Any] = {
        "cpu_temp_c": None,
        "gpu_temp_c": None,
        "gpu_memory_used_gb": None,
        "gpu_memory_total_gb": None,
    }

    # Try tegrastats first (always available on Jetson)
    try:
        import subprocess
        result = subprocess.run(
            ["tegrastats", "--count", "1", "--interval", "100"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            metrics.update(_parse_tegrastats(result.stdout))
            return metrics
    except Exception:
        pass

    # Fallback: try jtop
    try:
        from jtop import jtop
        with jtop() as jetson:
            stats = jetson.stats
            if stats:
                # CPU temp from thermal zones
                if "Temp AO" in stats:
                    metrics["cpu_temp_c"] = float(stats["Temp AO"])
                if "Temp GPU" in stats:
                    metrics["gpu_temp_c"] = float(stats["Temp GPU"])
                # GPU memory
                if "RAM" in stats:
                    used = stats["RAM"].get("used", 0) / (1024**2)
                    total = stats["RAM"].get("total", 0) / (1024**2)
                    metrics["gpu_memory_used_gb"] = round(used, 1)
                    metrics["gpu_memory_total_gb"] = round(total, 1)
            return metrics
    except Exception:
        pass

    # Desktop fallback: try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 1:
                metrics["gpu_temp_c"] = float(parts[0].strip())
            if len(parts) >= 2:
                metrics["gpu_memory_used_gb"] = round(float(parts[1].strip()) / 1024, 1)
            if len(parts) >= 3:
                metrics["gpu_memory_total_gb"] = round(float(parts[2].strip()) / 1024, 1)
    except Exception:
        pass

    return metrics


def _parse_tegrastats(output: str) -> dict[str, Any]:
    """Parse tegrastats single-line output."""
    result: dict[str, Any] = {}
    try:
        # Example: "RAM 4785/15503MB ... CPU@43.5C GPU@42C ..."
        line = output.strip().split("\n")[-1]

        # GPU temp
        if "GPU@" in line:
            idx = line.index("GPU@") + 4
            end = line.index("C", idx)
            result["gpu_temp_c"] = float(line[idx:end])

        # CPU temp
        if "CPU@" in line:
            idx = line.index("CPU@") + 4
            end = line.index("C", idx)
            result["cpu_temp_c"] = float(line[idx:end])

        # RAM
        if "RAM " in line:
            idx = line.index("RAM ") + 4
            end = line.index("MB", idx)
            parts = line[idx:end].split("/")
            result["gpu_memory_used_gb"] = round(int(parts[0]) / 1024, 1)
            result["gpu_memory_total_gb"] = round(int(parts[1]) / 1024, 1)
    except Exception:
        pass
    return result
