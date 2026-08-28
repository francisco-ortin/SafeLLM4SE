"""Command-line entry point for the generic adaptive sampler."""
import argparse
import json

from sampling import SamplerSettings
from sampling.cli import parse_args, settings_from_args
from sampling.evaluators import load_evaluator, Evaluator
from sampling.sampler import AdaptiveSampler


def main() -> None:
    """Run the adaptive sampler command-line entry point.
    """
    print("Running SafeLLM4SE sampler...")
    args: argparse.Namespace = parse_args()
    settings: SamplerSettings = settings_from_args(args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator: Evaluator = load_evaluator(
        args.evaluator,
        settings.evaluator_parameters,
    )
    result: dict[str, object] = AdaptiveSampler(settings).run(evaluator)
    print(json.dumps(result, indent=2))
    print(f"Sample written in {settings.measurements_path}.")


if __name__ == "__main__":
    main()
