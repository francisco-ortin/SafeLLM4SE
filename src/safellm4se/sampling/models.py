"""Shared data structures for adaptive sampling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SamplingObservation:
    """One evaluator measurement."""
    theta: float  # Numeric theta observed by the evaluator.
    # Name of the experiment represented by the evaluator.
    experiment_name: str = ""
    model_name: str = ""  # Human-readable model family or provider label.
    model_id: str = ""  # Provider-specific model identifier used for the call.
    prompt_tokens: int = 0  # Number of input tokens consumed by the evaluation.
    completion_tokens: int = 0  # Number of output tokens produced by the evaluation.
    total_tokens: int = 0  # Total number of tokens consumed by the evaluation.
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra evaluator data.


@dataclass(frozen=True)
class SamplerSettings:
    """Runtime settings for one adaptive sampling run."""

    output_dir: Path  # Directory used for sampler output files.
    measurements_path: Path  # CSV file where completed observations are stored.
    reservations_path: Path  # JSON file with in-progress execution reservations.
    lock_path: Path  # File path used to coordinate concurrent sampler processes.
    task_id: str  # Identifier of the task being sampled.
    ci_method: str  # Confidence interval method requested for stopping checks.
    confidence_level: float  # Confidence level used for interval estimation.
    n_min: int  # Minimum observations required before stopping is allowed.
    target_ci_width: float  # Maximum interval width accepted by the stop criterion.
    budget_tokens: int  # Maximum total tokens allowed for the task and model.
    inter_invocation_waiting: float  # Delay between evaluator invocations in seconds.
    reservation_ttl_seconds: float  # Lifetime of unfinished execution reservations.
    run_id: str  # Unique identifier for this sampler process run.
    # Evaluator constructor arguments used for the sampling run.
    evaluator_parameters: dict[str, Any] = field(default_factory=dict)
