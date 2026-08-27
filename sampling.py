"""Command-line entry point for the generic adaptive sampler."""
import argparse
import json
from dataclasses import replace

from sampling import SamplerSettings
from sampling.cli import parse_args, settings_from_args
from sampling.evaluators import load_evaluator, Evaluator
from sampling.sampler import AdaptiveSampler


def main() -> None:
    args: argparse.Namespace = parse_args()
    settings: SamplerSettings = settings_from_args(args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator: Evaluator = load_evaluator(
        settings.evaluator_name,
        settings.evaluator_parameters,
    )
    settings = replace(settings, metric_type=evaluator.metric_type)
    result: dict[str, object] = AdaptiveSampler(settings).run(evaluator)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
