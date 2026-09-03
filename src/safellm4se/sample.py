"""Command-line entry point for the generic adaptive sampler."""
import argparse
import json
from pathlib import Path

from loguru import logger

from safellm4se.sampling import SamplerSettings
from safellm4se.sampling.config.logger import create_log_file_path, setup_logger
from safellm4se.sampling.cli import parse_args, settings_from_args
from safellm4se.sampling.evaluators import Evaluator, load_evaluator
from safellm4se.sampling.sampler import AdaptiveSampler


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
    """Run the adaptive sampler command-line entry point.

    Raises:
        Exception: Re-raises uncaught exceptions from evaluator loading or sampling.
    """
    args: argparse.Namespace = parse_args()
    settings: SamplerSettings = settings_from_args(args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    log_file: Path | None = setup_logger(
        create_log_file_path(settings.output_dir),
        console_level=_console_log_level_from_args(args),
    )
    logger.info("Running SafeLLM4SE sampler...")
    if log_file is not None:
        logger.info(f"Execution log written to {log_file}.")
    try:
        evaluator: Evaluator = load_evaluator(
            args.evaluator,
            settings.evaluator_parameters,
        )
        result: dict[str, object] = AdaptiveSampler(settings).run(evaluator)
        logger.info("Sampling result:\n{}", json.dumps(result, indent=2))
        final_msg = f"Sample written in {settings.measurements_path}."
        logger.info(final_msg)
        print(final_msg)
    except Exception:
        logger.exception("SafeLLM4SE sampler execution failed.")
        raise


if __name__ == "__main__":
    main()
