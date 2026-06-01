"""Console output backend with rich formatting."""

import time

from src.models.base import InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ANSI color codes for different event types
COLORS = {
    "kws": "\033[1;36m",  # Cyan bold
    "sed": "\033[1;33m",  # Yellow bold
    "asc": "\033[1;35m",  # Magenta bold
    "vad": "\033[0;32m",  # Green
    "reset": "\033[0m",
    "dim": "\033[2m",
}

# Unicode symbols for event types
SYMBOLS = {
    "kws": "🎤",
    "sed": "🔔",
    "asc": "🏠",
    "vad": "📢",
    "start": "▶",
    "stop": "⏹",
}


class ConsoleOutput:
    """Pretty-prints inference events to the console."""

    def __init__(self, color: bool = True):
        self._color = color
        self._start_time = time.time()

    def emit(self, result: InferenceResult) -> None:
        """Print an inference result to stdout.

        Args:
            result: InferenceResult to display.
        """
        elapsed = time.time() - self._start_time
        symbol = SYMBOLS.get(result.task, "•")
        color = COLORS.get(result.task, "")
        reset = COLORS["reset"] if self._color else ""
        dim = COLORS["dim"] if self._color else ""

        if result.task == "vad":
            # VAD state changes are subtle
            state_emoji = "🔊" if result.label == "speech" else "🔇"
            if self._color:
                line = (
                    f"{dim}[{elapsed:7.2f}s]{reset} "
                    f"{state_emoji} {color}VAD: {result.label}{reset} "
                    f"({result.confidence:.2f})"
                )
            else:
                line = f"[{elapsed:7.2f}s] VAD: {result.label} ({result.confidence:.2f})"
        elif result.task == "kws":
            if result.label == "no_keyword":
                return  # Don't spam for no-keyword
            line = (
                f"{symbol} {color}[{elapsed:7.2f}s] KWS: "
                f"'{result.label}' ({result.confidence:.2f}) "
                f"⏱ {result.latency_ms:.1f}ms{reset}"
            )
        elif result.task == "sed":
            line = (
                f"{symbol} {color}[{elapsed:7.2f}s] SED: "
                f"{result.label} ({result.confidence:.2f}) "
                f"⏱ {result.latency_ms:.1f}ms{reset}"
            )
        elif result.task == "asc":
            line = (
                f"{symbol} {color}[{elapsed:7.2f}s] ASC: "
                f"{result.label} ({result.confidence:.2f}) "
                f"⏱ {result.latency_ms:.1f}ms{reset}"
            )
        else:
            line = f"[{elapsed:7.2f}s] {result.task}: {result.label} ({result.confidence:.2f})"

        print(line, flush=True)

    def emit_status(self, message: str) -> None:
        """Print a status message (startup, shutdown, config changes)."""
        elapsed = time.time() - self._start_time
        print(f"  {COLORS['dim']}[{elapsed:7.2f}s] {message}{COLORS['reset']}")
