"""Auto-remediation actions for edge device.

When the system monitor detects an anomaly, these actions can be
triggered automatically or via the agent.
"""

from __future__ import annotations

import subprocess

from src.utils.logging import get_logger

logger = get_logger(__name__)


def clear_system_cache() -> bool:
    """Drop page cache and dentries (requires sudo on Linux).

    Safe — only drops clean caches, doesn't affect dirty pages.
    """
    try:
        subprocess.run(
            ["sync"],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, timeout=5, shell=False,
        )
        logger.info("System cache cleared")
        return True
    except Exception as e:
        logger.warning(f"Cache clear failed: {e}")
        return False


def restart_service(service_name: str) -> bool:
    """Restart a systemd service."""
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", service_name],
            capture_output=True, timeout=30,
        )
        logger.info(f"Service restarted: {service_name}")
        return True
    except Exception as e:
        logger.warning(f"Service restart failed ({service_name}): {e}")
        return False


def throttle_gpu(max_clock_mhz: int = 800) -> bool:
    """Set Jetson GPU max clock (requires sudo)."""
    try:
        subprocess.run(
            ["sudo", "sh", "-c", f"echo {max_clock_mhz} > /sys/devices/gpu.0/devfreq/57000000.ga10b/max_freq"],
            capture_output=True, timeout=5,
        )
        logger.info(f"GPU throttled to {max_clock_mhz} MHz")
        return True
    except Exception as e:
        logger.warning(f"GPU throttle failed: {e}")
        return False
