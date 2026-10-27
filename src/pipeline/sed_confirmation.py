"""Require consecutive windows of the same target before surfacing a candidate."""

from src.models.base import InferenceResult


class SEDConfirmation:
    def __init__(self, windows: int = 2):
        if windows < 1:
            raise ValueError("windows must be positive")
        self.windows = windows
        self._label = None
        self._count = 0

    def update(self, result: InferenceResult) -> bool:
        if result.label == "Silence":
            self._label, self._count = None, 0
            return False
        self._count = self._count + 1 if result.label == self._label else 1
        self._label = result.label
        return self._count >= self.windows
