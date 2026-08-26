"""Logging setup helpers for console and file output."""

import multiprocessing
import os
import sys
from datetime import datetime
from typing import Optional

from loguru import logger

from sampling.config.config import Config


def _create_new_log_file_name() -> str:
    """Create a timestamped log file path inside the log directory."""
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name: str = f"log_{timestamp}.log"

    logs_dir: str = Config.LOG_DIR
    os.makedirs(logs_dir, exist_ok=True)

    return os.path.join(logs_dir, file_name)


def setup_logger(
    console_level: str = "INFO",
    file_level: str = "TRACE",
    log_file: Optional[str] = None,
    auto_log_file: bool = False,
) -> Optional[str]:
    """Configure global console and optional file logging."""
    logger.remove()

    log_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "process:{process.name}:{process.id} | "
        "{name}:{function}:{line} - {message}"
    )

    logger.add(
        sys.stdout,
        level=console_level,
        format=log_format,
        colorize=True
    )

    is_main_process = multiprocessing.current_process().name == "MainProcess"

    if is_main_process and auto_log_file and not log_file:
        log_file = _create_new_log_file_name()

    if is_main_process and log_file:
        logger.add(
            log_file,
            level=file_level,
            format=log_format,
            encoding="utf-8",
            enqueue=True,
            rotation="10 MB",
            retention="1 month",
        )

    return log_file if is_main_process else None
