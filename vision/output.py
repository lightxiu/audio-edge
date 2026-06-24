"""Vision pipeline output — bridges camera + detector + tracker to the event bus.

``VisionOutput`` is the main entry point for the vision module.
It manages the camera → detect → track loop in a background thread
and emits structured event dicts.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from vision.camera import CameraCapture, MockCameraCapture
from vision.detector import Detection, YOLODetector
from vision.tracker import IoUTracker, Track
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VisionOutput:
    """Real-time vision pipeline output.

    Manages the camera → YOLO detection → object tracking loop in a
    background thread.  Emits structured event dicts via callbacks.

    Parameters
    ----------
    camera_id : int or str
        Camera device index or GStreamer pipeline string.
    model_path : str or Path
        Path to YOLO ONNX model.
    classes : list[str] or None
        Class names (default: COCO 80).
    confidence_threshold : float
        Detection confidence threshold.
    fps : int
        Target processing FPS.
    mock : bool
        Use mock camera (for development without a real camera).
    """

    def __init__(
        self,
        camera_id: int | str = 0,
        model_path: str | Path = "models/vision/yolov8n.onnx",
        classes: list[str] | None = None,
        confidence_threshold: float = 0.25,
        fps: int = 10,
        mock: bool = False,
    ) -> None:
        # Camera
        camera_cls = MockCameraCapture if mock else CameraCapture
        self._camera = camera_cls(fps=fps) if mock else camera_cls(camera_id=camera_id, fps=fps)

        # Detector
        self._detector = YOLODetector(
            model_path=model_path,
            classes=classes,
            confidence_threshold=confidence_threshold,
        )

        # Tracker
        self._tracker = IoUTracker()

        # State
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[dict], None]] = []
        self._latest_result: dict | None = None
        self._lock = threading.Lock()

    # -- Public API -------------------------------------------------------
    def start(self) -> None:
        """Open camera, load model, start background processing thread."""
        logger.info("Vision pipeline starting...")

        ok = self._camera.open()
        if not ok:
            raise RuntimeError(f"Failed to open camera {self._camera._camera_id}")

        self._detector.load()
        logger.info(
            f"Camera: {self._camera.width}x{self._camera.height} @ {self._camera.fps}fps"
        )
        logger.info(f"Detector: {'mock' if self._detector.is_mock else 'YOLO ONNX'}")

        self._running.set()
        self._thread = threading.Thread(
            target=self._loop,
            name="vision-pipeline",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background thread and release resources."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._camera.close()
        self._detector.unload()
        logger.info("Vision pipeline stopped.")

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for detection events.

        Callback receives a dict:
            {
                "module": "vision",
                "timestamp": float,
                "detections": [Detection, ...],
                "tracks": [Track, ...],
                "summary": {"person": 2, "car": 1, ...},
            }
        """
        self._callbacks.append(callback)

    def get_latest(self) -> dict | None:
        """Return the most recent detection result (thread-safe)."""
        with self._lock:
            return self._latest_result

    # -- Background loop --------------------------------------------------
    def _loop(self) -> None:
        """Main processing loop: read frame → detect → track → emit."""
        frame_interval = 1.0 / max(self._camera.fps, 1)

        while self._running.is_set():
            t_start = time.perf_counter()

            frame = self._camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            detections = self._detector.detect(frame)
            tracks = self._tracker.update(detections)

            # Build event dict
            summary = _build_summary(tracks) if tracks else _build_summary_from_dets(detections)
            result = {
                "module": "vision",
                "timestamp": time.time(),
                "detections": [_det_to_dict(d) for d in detections],
                "tracks": [_track_to_dict(t) for t in tracks],
                "summary": summary,
            }

            with self._lock:
                self._latest_result = result

            # Notify subscribers
            for cb in self._callbacks:
                try:
                    cb(result)
                except Exception as e:
                    logger.error(f"Vision callback error: {e}")

            # Throttle to target FPS
            elapsed = time.perf_counter() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)


# -- Helpers ----------------------------------------------------------------

def _det_to_dict(d: Detection) -> dict:
    return {
        "class_name": d.class_name,
        "confidence": round(d.confidence, 4),
        "box": list(d.box),
    }


def _track_to_dict(t: Track) -> dict:
    return {
        "track_id": t.track_id,
        "class_name": t.class_name,
        "confidence": round(t.confidence, 4),
        "box": list(t.box),
        "frames_seen": t.frames_seen,
        "duration_sec": round(t.last_seen - t.first_seen, 1),
    }


def _build_summary(tracks: list[Track]) -> dict[str, int]:
    """Count active tracks by class."""
    counts: dict[str, int] = {}
    for t in tracks:
        counts[t.class_name] = counts.get(t.class_name, 0) + 1
    return counts


def _build_summary_from_dets(detections: list[Detection]) -> dict[str, int]:
    """Count detections by class (used when no tracks are confirmed yet)."""
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    return counts
