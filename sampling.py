"""Command-line entry point for the generic adaptive sampler."""

import json
from dataclasses import replace

from sampling.cli import parse_args, settings_from_args
from sampling.evaluators import load_evaluator
from sampling.sampler import AdaptiveSampler


def main() -> None:
    args = parse_args()
    settings = settings_from_args(args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator(
        settings.evaluator_name,
        settings.evaluator_parameters,
    )
    settings = replace(settings, metric_type=evaluator.metric_type)
    result = AdaptiveSampler(settings).run(evaluator)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "summary_rows"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
