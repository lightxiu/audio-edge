"""audio-edge real-time WebSocket dashboard.

This package is self-contained — copy it to any project that needs
a real-time event dashboard. Only ``output.py`` depends on the
audio-edge ``InferenceResult`` dataclass; ``server.py`` is generic.

Usage (embedded in audio-edge pipeline):
    from dashboard.output import DashboardOutput
    output = DashboardOutput(port=8765)
    output.emit(result)  # pushes to all connected browsers

Usage (standalone, tailing a JSONL file):
    python -m dashboard.server --jsonl logs/events.jsonl --port 8765

Directory layout::

    dashboard/
        __init__.py
        server.py    # FastAPI app, WebSocket endpoint, event buffer
        output.py    # Pipeline output backend (DashboardOutput)
        static/
            index.html  # Single-file frontend
        README.md    # This file
"""
