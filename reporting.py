"""Command-line entry point for SafeLLM4SE sampling CSV reports."""

import argparse
from pathlib import Path

from loguru import logger

from reporting.cli import parse_args, settings_from_args
from reporting.models import ReportingSettings
from reporting.reporter import generate_report
from sampling.config.logger import create_log_file_path, setup_logger


def _console_log_level_from_args(args: argparse.Namespace) -> str | None:
    """Return the console log level requested by command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A log level for console output, or None when console logging is disabled.
    """
    if args.log_level:
        return args.log_level
    if args.verbose:
        return "INFO"
    return None


def main() -> None:
    """Run the sampling report command-line entry point.

    Raises:
        Exception: Re-raises uncaught exceptions from report generation.
    """
    args: argparse.Namespace = parse_args()
    settings: ReportingSettings = settings_from_args(args)
    log_directory: Path = settings.output_path.parent
    log_file: Path | None = setup_logger(
        create_log_file_path(log_directory),
        console_level=_console_log_level_from_args(args),
    )
    logger.info("Running SafeLLM4SE report generation...")
    if log_file is not None:
        logger.info(f"Execution log written to {log_file}.")
    try:
        generate_report(settings)
        final_message: str = f"Report written in {settings.output_path}."
        logger.info(final_message)
        print(final_message)
    except Exception:
        logger.exception("SafeLLM4SE report generation failed.")
        raise


if __name__ == "__main__":
    main()
