"""Inference task scheduler.

Determines when each inference task (KWS, SED, ASC) should run based on
audio activity and configured intervals.
"""

import time
from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskSchedule:
    """Scheduling state for a single inference task."""

    name: str
    interval_sec: float  # Minimum time between runs
    last_run: float = 0.0
    enabled: bool = True
    requires_speech: bool = False  # Only run when speech is active

    @property
    def due(self) -> bool:
        """Whether the task is due to run."""
        if not self.enabled:
            return False
        return (time.time() - self.last_run) >= self.interval_sec

    def mark_run(self) -> None:
        self.last_run = time.time()

    def reset(self) -> None:
        self.last_run = 0.0


@dataclass
class Scheduler:
    """Manages scheduling for multiple inference tasks.

    KWS: runs every ~100ms during speech segments
    SED: runs every ~1s on any audio
    ASC: runs every ~2-3s continuously
    """

    tasks: dict[str, TaskSchedule] = field(default_factory=dict)

    def add_task(
        self,
        name: str,
        interval_sec: float,
        requires_speech: bool = False,
    ) -> None:
        """Register a task with the scheduler."""
        self.tasks[name] = TaskSchedule(
            name=name,
            interval_sec=interval_sec,
            requires_speech=requires_speech,
        )
        logger.debug(f"Scheduler: registered task '{name}' (interval={interval_sec}s)")

    def is_due(self, name: str, speech_active: bool = True) -> bool:
        """Check if a task is due to run.

        Args:
            name: Task name.
            speech_active: Whether speech is currently detected.

        Returns:
            True if the task should run now.
        """
        if name not in self.tasks:
            return False

        task = self.tasks[name]
        if not task.enabled:
            return False

        if task.requires_speech and not speech_active:
            return False

        return task.due

    def mark_run(self, name: str) -> None:
        """Mark a task as having just run."""
        if name in self.tasks:
            self.tasks[name].mark_run()

    def get_due_tasks(self, speech_active: bool = True) -> list[str]:
        """Get list of task names that are due to run.

        Args:
            speech_active: Whether speech is currently detected.

        Returns:
            List of task names.
        """
        return [name for name in self.tasks if self.is_due(name, speech_active)]

    def reset(self) -> None:
        """Reset all task timers."""
        for task in self.tasks.values():
            task.reset()

    def disable(self, name: str) -> None:
        """Disable a task by name."""
        if name in self.tasks:
            self.tasks[name].enabled = False

    def enable(self, name: str) -> None:
        """Enable a task by name."""
        if name in self.tasks:
            self.tasks[name].enabled = True
