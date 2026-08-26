"""Default values used by sampling.py when CLI arguments are omitted."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration defaults for adaptive sampling runs."""

    output_dir: str = "output"
    measurements_file_name: str = "measurements.csv"
    results_file_name: str = "results.csv"
    reservations_file_name: str = ".sampling_reservations.json"
    lock_file_name: str = ".sampling.lock"

    task_id: str = "default-task"
    prompt: str = ""
    model: str = "random-binary"
    metric_type: str = "binary"
    ci_method: str = "auto"

    temperature: float = 0.2
    confidence_level: float = 0.95
    n_min: int = 30
    target_ci_width: float = 0.10
    budget: int = 100
    bootstrap_samples: int = 2000
    max_tokens: int = 256
    inter_invocation_waiting: float = 0.0
    reservation_ttl_seconds: float = 24 * 60 * 60

    gemini_model: str = "gemini-3.1-flash-lite"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_host: str = "http://127.0.0.1:11434"
    api_keys_file: str = "safe/api-keys.json"
    system_prompt: str = (
        "Return only a numeric quality score for the requested output."
    )


config = SamplingConfig()
