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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.output.review import ReviewStore

# ------------------------------------------------------------------
# Thread-safe event buffer
# ------------------------------------------------------------------

_event_lock = threading.Lock()
_event_buffer: deque[dict] = deque(maxlen=500)

# Store startup time for uptime display
_start_time = time.time()
_review_store: ReviewStore | None = None


def configure_review_store(store: ReviewStore | None) -> None:
    global _review_store
    _review_store = store


def _require_review_store() -> ReviewStore:
    if _review_store is None:
        raise HTTPException(status_code=404, detail="Review is not enabled")
    return _review_store


class ReviewDecision(BaseModel):
    status: str
    note: str = Field(default="", max_length=500)


class VideoObject(BaseModel):
    label: int
    name: str = Field(max_length=120)
    probability: float = Field(ge=0, le=1)
    box: tuple[float, float, float, float]


class VideoEvent(BaseModel):
    type: str = Field(min_length=1, max_length=120)
    timestamp: int = Field(ge=946684800000)  # Unix epoch milliseconds, year 2000 onward
    count: int = Field(ge=0)
    objects: list[VideoObject] = Field(max_length=100)
    message: str = Field(default="", max_length=1000)


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


@app.post("/api/video-events", status_code=201)
def ingest_video_event(event: VideoEvent, source_id: str = "unspecified"):
    store = _require_review_store()
    if len(source_id) > 120 or not source_id.strip():
        raise HTTPException(status_code=422, detail="Invalid source_id")
    if event.count != len(event.objects):
        raise HTTPException(status_code=422, detail="count must equal objects length")
    payload = event.model_dump() if hasattr(event, "model_dump") else event.dict()
    return store.add_video_event(payload, source_id)


@app.get("/api/video-events")
def list_video_events():
    return _require_review_store().list_video_events()


@app.get("/api/associations")
def list_associations(window_sec: float = 5.0):
    if not 0 <= window_sec <= 60:
        raise HTTPException(status_code=422, detail="window_sec must be between 0 and 60")
    store = _require_review_store()
    return [
        {
            "audio": audio,
            "nearby_video": store.nearby_video_events(audio["timestamp"], window_sec),
            "media_evidence": store.get_media_evidence(audio["id"]),
            "relationship": "same_media_file" if store.get_media_evidence(audio["id"]) else "temporal_only",
        }
        for audio in store.list()
    ]


@app.get("/api/candidates")
def list_candidates(status: str | None = None):
    store = _require_review_store()
    try:
        return store.list(status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/candidates/{candidate_id}/audio")
def candidate_audio(candidate_id: str):
    store = _require_review_store()
    candidate = store.get(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    # The filename is generated internally, never constructed from the URL.
    return FileResponse(store.evidence_dir / candidate["evidence"], media_type="audio/wav")


@app.get("/api/candidates/{candidate_id}/video")
def candidate_video(candidate_id: str):
    store = _require_review_store()
    media = store.get_media_evidence(candidate_id)
    if media is None:
        raise HTTPException(status_code=404, detail="Video evidence not found")
    return FileResponse(store.evidence_dir / media["video_file"], media_type="video/mp4")


@app.patch("/api/candidates/{candidate_id}")
def review_candidate(candidate_id: str, decision: ReviewDecision):
    store = _require_review_store()
    try:
        candidate = store.review(candidate_id, decision.status, decision.note)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


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
