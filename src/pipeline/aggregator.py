"""Event aggregation — deduplication, throttling, and merging.

Prevents event storms by applying cooldown windows and hysteresis.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

from src.models.base import InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AggregatorConfig:
    """Configuration for event aggregation."""

    kws_cooldown_sec: float = 1.5  # Min time between same keyword
    sed_cooldown_sec: float = 3.0  # Min time between same sound event
    asc_hysteresis_sec: float = 2.0  # Min time before scene change confirmed


@dataclass
class EventAggregator:
    """Filters and deduplicates inference results into output events.

    Applies per-task cooldown logic:
      - KWS: Cooldown per keyword to avoid repeated triggers.
      - SED: Per-class cooldown (e.g., siren 5s, gunshot 2s).
      - ASC: Hysteresis — only report scene change after it persists.
    """

    config: AggregatorConfig = field(default_factory=AggregatorConfig)

    # Per-keyword last-trigger timestamps
    _kws_last_fire: dict[str, float] = field(default_factory=dict)
    # Per-sound-event last-trigger timestamps
    _sed_last_fire: dict[str, float] = field(default_factory=dict)
    # Current scene tracking
    _current_scene: str = "unknown"
    _scene_since: float = 0.0
    # Per-class SED cooldown overrides
    _sed_cooldowns: dict[str, float] = field(default_factory=dict)

    def should_emit(self, result: InferenceResult) -> bool:
        """Check whether an inference result should be emitted as an event.

        Args:
            result: Inference result from a model.

        Returns:
            True if the result should be passed to outputs.
        """
        if result.task == "kws":
            return self._check_kws(result)
        elif result.task == "sed":
            return self._check_sed(result)
        elif result.task == "asc":
            return self._check_asc(result)
        elif result.task == "vad":
            return True  # VAD state changes always pass through
        return True

    def _check_kws(self, result: InferenceResult) -> bool:
        """KWS deduplication: cooldown per keyword."""
        if result.label == "no_keyword":
            return False

        now = time.time()
        last = self._kws_last_fire.get(result.label, 0)
        if (now - last) < self.config.kws_cooldown_sec:
            return False

        self._kws_last_fire[result.label] = now
        return True

    def _check_sed(self, result: InferenceResult) -> bool:
        """SED deduplication: per-class cooldown."""
        label = result.label
        now = time.time()

        cooldown = self._sed_cooldowns.get(label, self.config.sed_cooldown_sec)
        last = self._sed_last_fire.get(label, 0)

        if (now - last) < cooldown:
            return False

        self._sed_last_fire[label] = now
        return True

    def _check_asc(self, result: InferenceResult) -> bool:
        """ASC hysteresis: only report scene change after persistence."""
        label = result.label
        now = time.time()

        if label == self._current_scene:
            return False  # No change

        # Check if the new scene has persisted long enough
        if self._scene_since == 0:
            # First time seeing this scene — start timer
            self._scene_since = now
            return False

        if (now - self._scene_since) >= self.config.asc_hysteresis_sec:
            old_scene = self._current_scene
            self._current_scene = label
            self._scene_since = 0
            logger.info(f"Scene changed: {old_scene} → {label}")
            return True

        return False

    def set_sed_cooldown(self, event_class: str, cooldown_sec: float) -> None:
        """Set a custom cooldown for a specific SED class.

        Example: set_sed_cooldown("siren", 5.0)  # Sirens have 5s cooldown
        """
        self._sed_cooldowns[event_class] = cooldown_sec

    def reset(self) -> None:
        """Reset all aggregation state."""
        self._kws_last_fire.clear()
        self._sed_last_fire.clear()
        self._current_scene = "unknown"
        self._scene_since = 0.0
