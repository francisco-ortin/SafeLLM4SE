"""Adaptive sampling algorithm."""

import math
import statistics
import time
from collections.abc import Callable

from sampling.evaluators import coerce_observation
from sampling.locking import interprocess_file_lock
from sampling.models import SamplerSettings
from sampling.persistence import (
    append_measurement_process_safe,
    create_measurement_row,
    read_current_values,
    remove_reservation,
    reserve_execution_number,
    summarize,
)
from sampling.statistics import confidence_interval


class AdaptiveSampler:
    """Adaptive sampler following the algorithm described in paper-v02.tex."""

    def __init__(self, settings: SamplerSettings) -> None:
        if settings.budget < settings.n_min:
            raise ValueError("budget must be greater than or equal to n_min.")
        self.settings = settings

    def should_stop(self, values: list[float]) -> bool:
        if len(values) < self.settings.n_min:
            return False
        ci_low, ci_high, _ = confidence_interval(
            values,
            self.settings.metric_type,
            self.settings.confidence_level,
            self.settings.ci_method,
            self.settings.bootstrap_samples,
        )
        return (ci_high - ci_low) <= self.settings.target_ci_width

    def run(self, evaluator: Callable[..., object]) -> dict[str, object]:
        while True:
            values = read_current_values(self.settings)
            if len(values) >= self.settings.budget or self.should_stop(values):
                break

            execution_number = reserve_execution_number(self.settings)
            if execution_number is None:
                break
            if self.settings.inter_invocation_waiting > 0:
                time.sleep(self.settings.inter_invocation_waiting)
            try:
                raw = evaluator(
                    prompt=self.settings.prompt,
                    task_id=self.settings.task_id,
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    execution_number=execution_number,
                )
                observation = coerce_observation(raw)
                row = create_measurement_row(
                    self.settings,
                    execution_number,
                    observation,
                )
                append_measurement_process_safe(self.settings, row)
            except Exception:
                with interprocess_file_lock(self.settings.lock_path):
                    remove_reservation(self.settings, execution_number)
                raise

        summary_rows = summarize(self.settings)
        values = read_current_values(self.settings)
        ci_low, ci_high, ci_method = confidence_interval(
            values,
            self.settings.metric_type,
            self.settings.confidence_level,
            self.settings.ci_method,
            self.settings.bootstrap_samples,
        )
        return {
            "task_id": self.settings.task_id,
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "n": len(values),
            "theta_hat": statistics.fmean(values) if values else math.nan,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width": ci_high - ci_low,
            "ci_method": ci_method,
            "measurements_path": str(self.settings.measurements_path),
            "results_path": str(self.settings.results_path),
            "summary_rows": summary_rows,
        }
