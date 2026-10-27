"""Pipeline output backend that pushes events to the WebSocket dashboard.

``DashboardOutput`` is a drop-in output backend — it implements the
same ``emit(result)`` / ``close()`` protocol as ``ConsoleOutput`` and
``JSONLOutput``, so the orchestrator can use it without any special
handling.

When instantiated, it starts a uvicorn server in a background daemon
thread.  Events pushed via ``emit()`` land in the server's shared
buffer and are broadcast to all connected browsers.

This is the **only** file in ``dashboard/`` that depends on audio-edge
data types (``InferenceResult``).  The rest of the package is generic.
"""

from __future__ import annotations

import threading
from typing import Any

from src.models.base import InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DashboardOutput:
    """Real-time WebSocket dashboard output backend.

    Starts a FastAPI + uvicorn server in a background thread on first
    ``emit()`` call (lazy start).  Pushes ``InferenceResult`` events
    to all connected browsers via WebSocket.

    Parameters
    ----------
    host : str
        Bind address (default ``"0.0.0.0"`` — accessible from LAN).
    port : int
        Listen port (default ``8765``).
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, review_store=None) -> None:
        if review_store is not None and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Evidence review dashboard must bind to localhost")
        self._host = host
        self._port = port
        self._server_thread: threading.Thread | None = None
        self._server = None
        self._started = False
        self._review_store = review_store

    # -- Output backend protocol ---------------------------------------

    def emit(self, result: InferenceResult) -> None:
        """Push one inference result to the dashboard WebSocket."""
        self._ensure_server()
        from dashboard.server import push_event

        # Include metadata if present (e.g. class_index, threshold, scenes)
        extra: dict[str, Any] = {}
        if result.metadata:
            extra["metadata"] = result.metadata

        push_event({
            "task": result.task,
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "timestamp": result.timestamp,
            "latency_ms": round(result.latency_ms, 2),
            **extra,
        })

    def close(self) -> None:
        """Shut down the uvicorn server gracefully."""
        if self._server_thread is not None and self._server_thread.is_alive():
            logger.info("Dashboard server stopping...")
            self._server.should_exit = True
            self._server_thread.join(timeout=2.0)

    def start(self) -> None:
        """Make the review page available before the first detection."""
        self._ensure_server()

    # -- Internal ------------------------------------------------------

    def _ensure_server(self) -> None:
        """Lazily start uvicorn in a background daemon thread."""
        if self._started:
            return
        self._started = True

        import uvicorn

        from dashboard.server import app, configure_review_store
        configure_review_store(self._review_store)

        config = uvicorn.Config(
            app=app,
            host=self._host,
            port=self._port,
            log_level="info",
            # Access log off — pipeline already logs events
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._server = server

        def _run():
            server.run()

        self._server_thread = threading.Thread(
            target=_run,
            name="dashboard-uvicorn",
            daemon=True,
        )
        self._server_thread.start()

        # Wait for server to actually be listening (max 5 seconds)
        import time
        import urllib.request

        display_host = "localhost" if self._host == "0.0.0.0" else self._host
        health_url = f"http://{display_host}:{self._port}/health"

        for _ in range(50):
            try:
                urllib.request.urlopen(health_url, timeout=0.5)
                break
            except Exception:
                time.sleep(0.1)

        logger.info(f"Dashboard ready → http://{display_host}:{self._port}")
