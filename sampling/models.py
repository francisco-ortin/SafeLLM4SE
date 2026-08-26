"""Shared data structures for adaptive sampling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SamplingObservation:
    """One evaluator measurement."""

    value: float
    passed: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplerSettings:
    """Runtime settings for one adaptive sampling run."""

    output_dir: Path
    measurements_path: Path
    results_path: Path
    reservations_path: Path
    lock_path: Path
    task_id: str
    prompt: str
    model: str
    evaluator_name: str
    metric_type: str
    ci_method: str
    temperature: float
    confidence_level: float
    n_min: int
    target_ci_width: float
    budget: int
    bootstrap_samples: int
    max_tokens: int
    inter_invocation_waiting: float
    reservation_ttl_seconds: float
    run_id: str
