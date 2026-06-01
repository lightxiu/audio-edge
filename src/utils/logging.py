"""Logging setup using loguru."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    fmt: str = "text",
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure loguru logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        fmt: Format style — "text" for human-readable, "json" for structured.
        log_file: Optional file path for log output.
    """
    # Remove default handler
    logger.remove()

    if fmt == "json":
        log_format = (
            '{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"name": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"}'
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Stderr handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=(fmt != "json"),
        backtrace=True,
        diagnose=True,
    )

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format=log_format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    logger.debug(f"Logging initialized: level={level}, fmt={fmt}")


def get_logger(name: str):
    """Get a logger bound with the given module name.

    In loguru, this is done via logger.bind() — but for simplicity we
    return the global logger since loguru handles module names automatically
    via __name__ in the format string.
    """
    return logger.bind(name=name)
