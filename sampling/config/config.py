"""Default values used by sampling.py when CLI arguments are omitted."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration defaults for adaptive sampling runs."""

    output_dir: str = "output"
    measurements_file_name: str = "measurements.csv"
    reservations_file_name: str = ".sampling_reservations.json"
    lock_file_name: str = ".sampling.lock"

    ci_method: str = "auto"

    confidence_level: float = 0.95
    n_min: int = 30
    target_ci_width: float = 0.10
    budget_tokens: int = 10_000
    bootstrap_samples: int = 2_000
    max_tokens: int = 256
    inter_invocation_waiting: float = 0.0
    reservation_ttl_seconds: float = 1 * 60 * 60  # one hour

    api_keys_file: str = "myevaluators/api-keys.json"
    system_prompt: str = (
        "Return only a numeric quality score for the requested output."
    )


config = SamplingConfig()
