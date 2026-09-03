"""Command-line entry point for SafeLLM4SE two-sample CSV comparisons."""

import argparse
from pathlib import Path

from loguru import logger

from safellm4se.comparing.cli import parse_args, settings_from_args
from safellm4se.comparing.comparator import generate_comparison_report
from safellm4se.comparing.models import ComparisonSettings
from safellm4se.sampling.config.logger import create_log_file_path, setup_logger


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
    """Run the comparison report command-line entry point.

    Raises:
        Exception: Re-raises uncaught exceptions from comparison generation.
    """
    args: argparse.Namespace = parse_args()
    settings: ComparisonSettings = settings_from_args(args)
    log_directory: Path = settings.output_path.parent
    log_file: Path | None = setup_logger(
        create_log_file_path(log_directory),
        console_level=_console_log_level_from_args(args),
    )
    logger.info("Running SafeLLM4SE comparison report generation...")
    if log_file is not None:
        logger.info(f"Execution log written to {log_file}.")
    try:
        generate_comparison_report(settings)
        final_message: str = f"Comparison report written in {settings.output_path}."
        logger.info(final_message)
        print(final_message)
    except Exception:
        logger.exception("SafeLLM4SE comparison report generation failed.")
        raise


if __name__ == "__main__":
    main()
