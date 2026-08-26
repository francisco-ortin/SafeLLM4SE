"""Command-line parsing for the adaptive sampler."""

import argparse
from pathlib import Path
from typing import Any
from uuid import uuid4

from sampling.config import config
from sampling.models import SamplerSettings

_MISSING = object()


PARAMETER_HELP: dict[str, str] = {
    "output_dir": "Directory where measurements, summaries, locks, and reservations are stored.",
    "task_id": "Identifier of the evaluation task or prompt being sampled.",
    "prompt": "Prompt text sent to the evaluator. Use --prompt-file for longer prompts.",
    "prompt_file": "Path to a UTF-8 text file whose contents replace --prompt.",
    "model": "Model name recorded in the output files and passed to the evaluator.",
    "evaluator": "Evaluator callable to execute: a built-in name or a module:callable reference.",
    "metric_type": "Metric family used to select the confidence interval calculation.",
    "ci_method": "Confidence interval method for continuous metrics; binary metrics always use Wilson.",
    "temperature": "Sampling temperature recorded in the output and passed to LLM evaluators.",
    "confidence_level": "Confidence level used for confidence intervals.",
    "n_min": "Minimum number of observations collected before checking the stopping criterion.",
    "target_ci_width": "Maximum total confidence interval width required to stop sampling.",
    "budget": "Maximum number of executions allowed for the task/model/temperature key.",
    "bootstrap_samples": "Number of bootstrap resamples used when --ci-method is bootstrap.",
    "max_tokens": "Maximum number of tokens requested from LLM evaluators.",
    "inter_invocation_waiting": "Seconds to wait before each evaluator invocation in this process.",
    "reservation_ttl_seconds": "Seconds after which unfinished execution reservations can be reused.",
}


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAFE-style adaptive sampling framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_argument(parser, "evaluator", "--evaluator")
    _add_argument(parser, "output_dir", "--output-dir")
    _add_argument(parser, "task_id", "--task-id")
    _add_argument(parser, "prompt", "--prompt")
    parser.add_argument("--prompt-file", help=PARAMETER_HELP["prompt_file"])
    _add_argument(parser, "model", "--model")
    _add_argument(
        parser,
        "metric_type",
        "--metric-type",
        choices=["binary", "continuous"],
    )
    _add_argument(
        parser,
        "ci_method",
        "--ci-method",
        choices=["auto", "t", "bootstrap"],
    )
    _add_argument(parser, "temperature", "--temperature", type=float)
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
        type=float,
    )
    _add_argument(
        parser,
        "reservation_ttl_seconds",
        "--reservation-ttl-seconds",
        type=float,
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> SamplerSettings:
    output_dir = Path(args.output_dir)
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    model = args.model
    if model == _config_value("model") and args.evaluator in {"gemini", "gemini_quality"}:
        model = _config_value("gemini_model")
    if model == _config_value("model") and args.evaluator in {"ollama", "ollama_quality"}:
        model = _config_value("ollama_model")

    return SamplerSettings(
        output_dir=output_dir,
        measurements_path=output_dir / _required_config_value("measurements_file_name"),
        results_path=output_dir / _required_config_value("results_file_name"),
        reservations_path=output_dir / _required_config_value("reservations_file_name"),
        lock_path=output_dir / _required_config_value("lock_file_name"),
        task_id=args.task_id,
        prompt=prompt,
        model=model,
        evaluator_name=args.evaluator,
        metric_type=args.metric_type,
        ci_method=args.ci_method,
        temperature=args.temperature,
        confidence_level=args.confidence_level,
        n_min=args.n_min,
        target_ci_width=args.target_ci_width,
        budget=args.budget,
        bootstrap_samples=args.bootstrap_samples,
        max_tokens=args.max_tokens,
        inter_invocation_waiting=args.inter_invocation_waiting,
        reservation_ttl_seconds=args.reservation_ttl_seconds,
        run_id=str(uuid4()),
    )


def _add_argument(
    parser: argparse.ArgumentParser,
    config_name: str,
    flag: str,
    **kwargs: Any,
) -> None:
    default = _config_value(config_name)
    kwargs.setdefault("help", PARAMETER_HELP[config_name])
    if default is _MISSING:
        kwargs["required"] = True
    else:
        kwargs["default"] = default
    parser.add_argument(flag, **kwargs)


def _config_value(name: str) -> Any:
    return getattr(config, name, _MISSING)


def _required_config_value(name: str) -> Any:
    value = _config_value(name)
    if value is _MISSING:
        raise AttributeError(f"config.{name} is required but is not defined.")
    return value
