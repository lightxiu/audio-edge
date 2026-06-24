"""USB / CSI camera capture via OpenCV.

Provides a unified interface for camera capture with mock fallback
for development / testing on machines without a physical camera.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class CameraCapture:
    """OpenCV-based camera capture.

    Parameters
    ----------
    camera_id : int or str
        Device index (0, 1, ...) or path like ``/dev/video0`` or
        a GStreamer pipeline string for CSI cameras on Jetson.
    width : int
        Capture width in pixels.
    height : int
        Capture height in pixels.
    fps : int
        Target frames per second.
    """

    def __init__(
        self,
        camera_id: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ) -> None:
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: object | None = None  # cv2.VideoCapture
        self._is_opened = False

    # -- Context manager --------------------------------------------------
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # -- Public API -------------------------------------------------------
    def open(self) -> bool:
        """Open the camera device. Returns True on success."""
        import cv2

        cap = cv2.VideoCapture(self._camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not cap.isOpened():
            cap.release()
            return False

        self._cap = cap
        self._is_opened = True
        return True

    def read(self) -> np.ndarray | None:
        """Read one frame. Returns BGR numpy array (H, W, 3) or None on failure."""
        if not self._is_opened or self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def close(self) -> None:
        """Release the camera."""
        self._is_opened = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- Properties -------------------------------------------------------
    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> int:
        return self._fps


class MockCameraCapture(CameraCapture):
    """Mock camera that generates synthetic frames for testing.

    Produces solid-color frames that cycle through colours over time,
    useful for verifying the pipeline without a physical camera.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15) -> None:
        super().__init__(camera_id=-1, width=width, height=height, fps=fps)
        self._frame_count = 0

    def open(self) -> bool:
        self._is_opened = True
        return True

    def read(self) -> np.ndarray | None:
        """Generate a synthetic frame that cycles through colours."""
        if not self._is_opened:
            return None
        self._frame_count += 1

        # Cycle hue over time so mock frames look "alive"
        hue = (self._frame_count % 180) / 180.0
        import cv2

        hsv = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        hsv[:, :, 0] = int(hue * 179)
        hsv[:, :, 1] = 128
        hsv[:, :, 2] = 192
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def close(self) -> None:
        self._is_opened = False
