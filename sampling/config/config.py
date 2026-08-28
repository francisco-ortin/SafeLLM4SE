"""Default values used by sampling.py when CLI arguments are omitted."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration defaults for adaptive sampling runs."""

    output_dir: str = "output"  # Directory where sampling outputs are stored.
    measurements_file_name: str = "measurements.csv"  # File name for collected measurements.
    reservations_file_name: str = ".sampling_reservations.json"  # File name for sampling reservations.
    lock_file_name: str = ".sampling.lock"  # File name used to coordinate concurrent sampling runs.

    ci_method: str = "auto"  # Confidence interval method used during evaluation.

    confidence_level: float = 0.95  # Target confidence level for statistical intervals.
    n_min: int = 10  # Minimum number of samples required before stopping checks.
    temperature: float = 0.2  # Default model sampling temperature.
    target_ci_width: float = 0.10  # Desired maximum confidence interval width.
    budget_tokens: int = 10_000  # Maximum token budget available for sampling.
    bootstrap_samples: int = 2_000  # Number of bootstrap resamples used for interval estimation.
    max_tokens: int = 256  # Maximum number of tokens requested per model invocation.
    inter_invocation_waiting: float = 0.0  # Delay in seconds between model invocations.
    reservation_ttl_seconds: float = 1 * 60 * 60  # Time in seconds before a reservation expires.

    api_keys_file: str = "myevaluators/api-keys.json"  # Path to the API keys configuration file.
    system_prompt: str = (
        "Return only a numeric quality score for the requested output."
    )  # Default system prompt used for scoring requests.


config = SamplingConfig()
