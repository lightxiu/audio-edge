"""Simple IoU-based multi-object tracker.

Tracks objects across frames by matching detections via IoU overlap.
Lighter than ByteTrack / DeepSORT — suitable for edge deployment
where every MB of memory counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Track:
    """A single tracked object."""

    track_id: int
    class_name: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    first_seen: float  # epoch seconds
    last_seen: float
    frames_seen: int = 1
    frames_missed: int = 0


class IoUTracker:
    """Simple IoU-based multi-object tracker.

    Parameters
    ----------
    max_missed : int
        How many frames a track can go unmatched before being removed.
    min_hits : int
        Minimum consecutive frames before a track is considered "confirmed".
    iou_threshold : float
        Minimum IoU to consider two boxes as the same object.
    """

    def __init__(
        self,
        max_missed: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self._max_missed = max_missed
        self._min_hits = min_hits
        self._iou_thresh = iou_threshold
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list) -> list[Track]:
        """Update tracks with new detections.

        Parameters
        ----------
        detections : list[Detection]
            Detections from the current frame (``Detection`` named tuples
            with ``class_name``, ``confidence``, ``box`` fields).

        Returns
        -------
        list[Track]
            All active tracks after this update.
        """
        now = time.time()

        # Age all tracks
        for t in self._tracks.values():
            t.frames_missed += 1

        # Match detections to existing tracks by IoU
        matched_track_ids: set[int] = set()

        for det in detections:
            best_id = None
            best_iou = self._iou_thresh

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                if track.class_name != det.class_name:
                    continue
                iou = self._box_iou(track.box, det.box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                # Update existing track
                t = self._tracks[best_id]
                t.box = det.box
                t.confidence = det.confidence
                t.last_seen = now
                t.frames_seen += 1
                t.frames_missed = 0
                matched_track_ids.add(best_id)
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(
                    track_id=tid,
                    class_name=det.class_name,
                    box=det.box,
                    confidence=det.confidence,
                    first_seen=now,
                    last_seen=now,
                )

        # Remove stale tracks
        stale = [
            tid for tid, t in self._tracks.items()
            if t.frames_missed > self._max_missed
        ]
        for tid in stale:
            del self._tracks[tid]

        return self.active_tracks

    # -- Queries ----------------------------------------------------------
    @property
    def active_tracks(self) -> list[Track]:
        """Return confirmed tracks (those that hit the min_hits threshold)."""
        return [
            t for t in self._tracks.values()
            if t.frames_seen >= self._min_hits
        ]

    @property
    def all_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    # -- Helpers ----------------------------------------------------------
    @staticmethod
    def _box_iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
