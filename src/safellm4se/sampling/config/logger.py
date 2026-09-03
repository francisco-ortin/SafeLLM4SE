"""Logging setup helpers for the adaptive sampler command-line program."""

import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


LOG_LEVELS: tuple[str, ...] = (
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
)


def create_log_file_path(output_dir: Path) -> Path:
    """Create a timestamped log file path inside the sampling output directory.

    Args:
        output_dir: Directory where sampling outputs are stored.

    Returns:
        Path to the log file for the current execution.
    """
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir: Path = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"log_{timestamp}.log"


def setup_logger(
    log_file: Path,
    console_level: str | None = None,
    file_level: str = "TRACE",
) -> Path | None:
    """Configure global logging for file output and optional console output.

    Args:
        log_file: File where execution logs are persisted.
        console_level: Optional minimum level shown in the console.
        file_level: Minimum level persisted in the log file.

    Returns:
        The active log file path in the main process, or None in child processes.

    Raises:
        OSError: If the log directory or log file cannot be created.
        ValueError: If Loguru rejects a configured log level.
    """
    logger.remove()

    log_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "process:{process.name}:{process.id} | "
        "{name}:{function}:{line} - {message}"
    )
    normalized_file_level: str = file_level.upper()
    normalized_console_level: str | None = (
        console_level.upper() if console_level else None
    )
    is_main_process: bool = multiprocessing.current_process().name == "MainProcess"

    if is_main_process:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=normalized_file_level,
            format=log_format,
            encoding="utf-8",
            enqueue=True,
            rotation="10 MB",
            retention="1 month",
        )

    if normalized_console_level:
        logger.add(
            sys.stdout,
            level=normalized_console_level,
            format=log_format,
            colorize=True,
        )

    return log_file if is_main_process else None
