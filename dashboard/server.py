"""FastAPI WebSocket server for the real-time audio-edge dashboard.

Architecture
------------
A thread-safe ``deque`` buffer bridges the pipeline threads (which call
``push_event()``) and the asyncio event loop (which serves WebSocket
clients).  No external message broker is needed.

Generic design
--------------
``push_event(data: dict)`` is the only public API.  It accepts any
JSON-serializable dict — no dependency on audio-edge data models.
This makes ``server.py`` reusable across projects (YOLO, etc.).

Endpoint summary
----------------
``GET  /``        → dashboard HTML page
``GET  /health``  → ``{"status":"ok"}`` health check
``WS   /ws``      → real-time event stream (JSON per line)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ------------------------------------------------------------------
# Thread-safe event buffer
# ------------------------------------------------------------------

_event_lock = threading.Lock()
_event_buffer: deque[dict] = deque(maxlen=500)

# Store startup time for uptime display
_start_time = time.time()


def push_event(data: dict) -> None:
    """Push one event into the buffer (thread-safe).

    Values that are not JSON-serializable (numpy scalars, etc.) are
    coerced to plain Python types automatically.
    """
    # Coerce any numpy / non-JSON values
    safe = {}
    for k, v in data.items():
        if isinstance(v, dict):
            safe[k] = {str(ik): _coerce(iv) for ik, iv in v.items()}
        else:
            safe[k] = _coerce(v)
    safe["server_ts"] = time.time()
    with _event_lock:
        _event_buffer.append(safe)


def _coerce(value):
    """Convert numpy scalars and other non-JSON types to plain Python."""
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return str(value)


def _drain_since(index: int) -> list[dict]:
    """Return events added since *index* (thread-safe)."""
    with _event_lock:
        return list(_event_buffer)[index:]


# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="audio-edge dashboard",
    docs_url=None,
    redoc_url=None,
)


@app.get("/health")
async def health():
    """Liveness probe — returns uptime and event count."""
    with _event_lock:
        count = len(_event_buffer)
    return {
        "status": "ok",
        "uptime_sec": round(time.time() - _start_time, 1),
        "events_buffered": count,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the single-page dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard/index.html not found</h1>", status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Push buffered events to a single browser client.

    Polls ``_event_buffer`` every 100 ms.  New clients receive only
    events that arrive *after* they connect.
    """
    await ws.accept()
    with _event_lock:
        seen = len(_event_buffer)  # skip events already in buffer
    try:
        while True:
            new_events = _drain_since(seen)
            for evt in new_events:
                await ws.send_json(evt)
            seen += len(new_events)
            await asyncio.sleep(0.1)  # poll every 100 ms
    except WebSocketDisconnect:
        pass  # browser closed — normal
    except Exception:
        import traceback
        traceback.print_exc()
