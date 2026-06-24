"""Alert notification dispatcher.

Routes alerts to configured channels: console log, WebSocket dashboard,
and (future) email/WeChat/webhook.
"""

from __future__ import annotations

import time
from typing import Callable

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global alert callbacks
_alert_callbacks: list[Callable[[dict], None]] = []


def register_alert_handler(callback: Callable[[dict], None]) -> None:
    """Register a function to be called on every alert.

    The callback receives a dict:
        {
            "module": "ops",
            "task": "alert",
            "label": str,
            "severity": "info" | "warning" | "critical",
            "timestamp": float,
            "metadata": dict,
        }
    """
    _alert_callbacks.append(callback)


def send_alert(
    message: str,
    severity: str = "warning",
    metadata: dict | None = None,
) -> None:
    """Send an alert to all registered handlers.

    Parameters
    ----------
    message : str
        Alert message (e.g. "GPU temperature exceeds 85°C").
    severity : str
        One of ``"info"``, ``"warning"``, ``"critical"``.
    metadata : dict or None
        Additional diagnostic data.
    """
    alert = {
        "module": "ops",
        "task": "alert",
        "label": message,
        "severity": severity,
        "timestamp": time.time(),
        "metadata": metadata or {},
    }

    # Always log
    log_fn = {
        "info": logger.info,
        "warning": logger.warning,
        "critical": logger.error,
    }.get(severity, logger.warning)
    log_fn(f"[{severity.upper()}] {message}")

    # Notify all registered handlers
    for cb in _alert_callbacks:
        try:
            cb(alert)
        except Exception as e:
            logger.error(f"Alert handler error: {e}")


def check_and_alert(status: dict) -> None:
    """Check system status and send alerts for any threshold violations.

    Parameters
    ----------
    status : dict
        Output from ``ops.monitor.get_system_status()``.
    """
    warnings = status.get("warnings", [])
    for w in warnings:
        # Determine severity
        severity = "warning"
        if "temp" in w and status.get("cpu_temp_c", 0) > 90:
            severity = "critical"
        elif "memory" in w and status.get("memory_percent", 0) > 95:
            severity = "critical"

        send_alert(w, severity=severity, metadata=status)
