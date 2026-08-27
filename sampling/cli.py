"""Command-line parsing for the adaptive sampler."""

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from sampling.config import config
from sampling.models import SamplerSettings

_MISSING = object()
_BOOLEAN_TEXT_VALUES: dict[str, bool] = {
    "false": False,
    "no": False,
    "true": True,
    "yes": True,
}
EVALUATOR_PARAMETERS_HELP: str = (
    "Evaluator constructor parameters:\n"
    "  Any arguments not recognized by the sampler are passed to the evaluator "
    "constructor after parsing.\n"
    "  Use the CLI separator followed by name=value pairs:\n"
    "    python sampling.py --evaluator sampling.myevaluators.random_binary_evaluator "
    "-- temperature=0.2 success_probability=0.7\n"
    "  The --name=value form is also accepted after the separator:\n"
    "    python sampling.py --evaluator sampling.myevaluators.random_binary_evaluator "
    "-- --temperature=0.2 --success-probability=0.7\n"
    "  Values are converted with Python literal syntax when possible, so numbers, "
    "booleans, None, lists, and dictionaries can be passed directly."
)


PARAMETER_HELP: dict[str, str] = {
    "output_dir": (
        "Directory where measurements, locks, and reservations are stored."
    ),
    "task_id": (
        "Identifier of the evaluation task being sampled. If omitted, the next "
        "tasks-id-<n> value is computed from the measurements CSV."
    ),
    "evaluator": (
        "Evaluator class to execute, using a module:ClassName reference or a "
        "module that defines exactly one concrete Evaluator class."
    ),
    "ci_method": (
        "Confidence interval method for continuous metrics; binary metrics always "
        "use Wilson."
    ),
    "confidence_level": "Confidence level used for confidence intervals.",
    "n_min": (
        "Minimum number of observations collected before checking the stopping "
        "criterion."
    ),
    "target_ci_width": (
        "Maximum total confidence interval width required to stop sampling."
    ),
    "budget": "Maximum number of evaluator invocations allowed for the task/model key.",
    "bootstrap_samples": (
        "Number of bootstrap resamples used when --ci-method is bootstrap."
    ),
    "max_tokens": "Maximum number of tokens requested from LLM evaluators.",
    "inter_invocation_waiting": (
        "Optional delay, in seconds, before each evaluator invocation made by this "
        "process. Use it to throttle providers with rate limits or to avoid sending "
        "bursts when several sampler processes run at the same time. A value of 0 "
        "disables the delay."
    ),
    "reservation_ttl_seconds": (
        "Optional time-to-live, in seconds, for an unfinished execution reservation. "
        "When a process reserves an execution number and crashes before writing its "
        "measurement, that reservation can be reused after this many seconds. Increase "
        "it for slow evaluators; decrease it to recover faster from interrupted runs."
    ),
}


class MandatoryAwareDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Formatter that shows mandatory arguments without reporting a None default."""

    def _get_help_string(self, action: argparse.Action) -> str:
        """Return the help text with mandatory status shown for required arguments."""

        help_text: str = action.help or ""
        if action.required:
            return f"{help_text} (mandatory)"
        return super()._get_help_string(action)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for an adaptive sampling run."""

    parser: argparse.ArgumentParser = build_parser()
    args, evaluator_parameter_tokens = parser.parse_known_args()
    args.evaluator_parameters = _parse_evaluator_parameters(
        evaluator_parameter_tokens,
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by the adaptive sampler."""

    parser = argparse.ArgumentParser(
        description="SafeLLM4SE adaptive sampling framework.",
        epilog=EVALUATOR_PARAMETERS_HELP,
        formatter_class=MandatoryAwareDefaultsHelpFormatter,
    )
    _add_argument(parser, "evaluator", "--evaluator", "--evaluador")
    _add_argument(parser, "output_dir", "--output-dir")
    parser.add_argument(
        "--task-id",
        default=argparse.SUPPRESS,
        help=PARAMETER_HELP["task_id"],
    )
    _add_argument(
        parser,
        "ci_method",
        "--ci-method",
        choices=["auto", "t", "bootstrap"],
    )
    _add_argument(parser, "confidence_level", "--confidence-level", type=float)
    _add_argument(parser, "n_min", "--n-min", type=int)
    _add_argument(parser, "target_ci_width", "--target-ci-width", type=float)
    _add_argument(parser, "budget", "--budget", type=int)
    _add_argument(parser, "bootstrap_samples", "--bootstrap-samples", type=int)
    _add_argument(parser, "max_tokens", "--max-tokens", type=int)
    _add_argument(
        parser,
        "inter_invocation_waiting",
        "--inter-invocation-waiting",
        "--inter_invocation_waiting",
        metavar="SECONDS",
        type=float,
    )
    _add_argument(
        parser,
        "reservation_ttl_seconds",
        "--reservation-ttl-seconds",
        "--reservation_ttl_seconds",
        metavar="SECONDS",
        type=float,
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> SamplerSettings:
    """Create sampler settings from parsed command-line arguments."""

    output_dir: Path = Path(args.output_dir)
    prompt: str = str(_config_value("prompt"))
    model: str = str(_config_value("model"))
    measurements_path: Path = (
        output_dir / _required_config_value("measurements_file_name")
    )
    provided_task_id: str | None = getattr(args, "task_id", None)
    task_id: str = provided_task_id or _next_task_id(measurements_path)

    return SamplerSettings(
        output_dir=output_dir,
        measurements_path=measurements_path,
        reservations_path=output_dir / _required_config_value("reservations_file_name"),
        lock_path=output_dir / _required_config_value("lock_file_name"),
        task_id=task_id,
        prompt=prompt,
        model=model,
        evaluator_name=args.evaluator,
        metric_type=str(_config_value("metric_type")),
        ci_method=args.ci_method,
        confidence_level=args.confidence_level,
        n_min=args.n_min,
        target_ci_width=args.target_ci_width,
        budget=args.budget,
        bootstrap_samples=args.bootstrap_samples,
        max_tokens=args.max_tokens,
        inter_invocation_waiting=args.inter_invocation_waiting,
        reservation_ttl_seconds=args.reservation_ttl_seconds,
        run_id=str(uuid4()),
        evaluator_parameters=args.evaluator_parameters,
    )


def _add_argument(
    parser: argparse.ArgumentParser,
    config_name: str,
    *flags: str,
    **kwargs: Any,
) -> None:
    """Add a CLI argument using project configuration defaults when present."""

    default: Any = _config_value(config_name)
    kwargs.setdefault("help", PARAMETER_HELP[config_name])
    if default is _MISSING:
        kwargs["required"] = True
    else:
        kwargs["default"] = default
    parser.add_argument(*flags, **kwargs)


def _config_value(name: str) -> Any:
    """Return a configuration value or the missing sentinel when absent."""

    return getattr(config, name, _MISSING)


def _required_config_value(name: str) -> Any:
    """Return a required configuration value or raise a clear error."""

    value: Any = _config_value(name)
    if value is _MISSING:
        raise AttributeError(f"config.{name} is required but is not defined.")
    return value


def _next_task_id(measurements_path: Path) -> str:
    """Return the next generated task identifier based on the measurements CSV."""

    highest_task_number: int = 0
    task_id_pattern: re.Pattern[str] = re.compile(r"^tasks-id-(\d+)$")
    if not measurements_path.exists() or measurements_path.stat().st_size == 0:
        return "tasks-id-1"

    with measurements_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader: csv.DictReader[str] = csv.DictReader(csv_file)
        for row in reader:
            task_id: str = str(row.get("task_id", ""))
            match: re.Match[str] | None = task_id_pattern.match(task_id)
            if match:
                highest_task_number = max(highest_task_number, int(match.group(1)))
    return f"tasks-id-{highest_task_number + 1}"


def _parse_evaluator_parameters(tokens: list[str]) -> dict[str, Any]:
    """Parse evaluator constructor parameters provided after the CLI separator."""

    parameters: dict[str, Any] = {}
    for token in tokens:
        if token == "--":
            continue
        if "=" not in token:
            raise ValueError(
                "Evaluator parameters must use name=value syntax after '--'."
            )
        name, raw_value = token.split("=", 1)
        clean_name: str = name.removeprefix("--").strip().replace("-", "_")
        if not clean_name:
            raise ValueError("Evaluator parameter names cannot be empty.")
        parameters[clean_name] = _coerce_parameter_value(raw_value.strip())
    return parameters


def _coerce_parameter_value(raw_value: str) -> Any:
    """Convert a CLI parameter value to the most suitable Python type."""

    normalized_value: str = raw_value.casefold()
    if normalized_value in _BOOLEAN_TEXT_VALUES:
        return _BOOLEAN_TEXT_VALUES[normalized_value]
    if normalized_value in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value
