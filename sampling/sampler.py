"""Adaptive sampling algorithm."""

import math
import statistics
import time

from sampling.config import config
from sampling.evaluators import Evaluator, run_evaluator
from sampling.locking import interprocess_file_lock
from sampling.models import SamplerSettings, SamplingObservation
from sampling.persistence import (
    append_measurement_process_safe,
    create_measurement_row,
    read_current_theta_and_total_tokens,
    remove_reservation,
    reserve_execution_number,
)
from sampling.statistics import confidence_interval


class AdaptiveSampler:
    """Adaptive sampler following the algorithm described in the SafeLLM4SE paper."""

    def __init__(self, settings: SamplerSettings) -> None:
        """Initialize the adaptive sampler.
        Args:
            settings: Sampling settings that define the task, budget, confidence
                interval configuration, and persistence paths.
        Raises:
            ValueError: If the token budget is not greater than zero.
        """
        if settings.budget_tokens <= 0:
            raise ValueError("budget_tokens must be greater than 0.")
        self.settings = settings

    def _ci_reached(self, theta_values: list[float], metric_type: str) -> bool:
        """Return whether adaptive sampling should stop.
        Args:
            theta_values: Theta observations already collected for the current
                task and model.
            metric_type: Type of metric used to select the confidence interval
                computation.
        Returns:
            True if the minimum sample size has been reached and the confidence
            interval width is within the target width; otherwise, False.
        """
        ci_low, ci_high, _ = confidence_interval(
            theta_values,
            metric_type,
            self.settings.confidence_level,
            self.settings.ci_method,
            config.bootstrap_samples,
        )
        return (ci_high - ci_low) <= self.settings.target_ci_width

    def run(self, evaluator: Evaluator) -> dict[str, object]:
        """Run adaptive sampling with the provided evaluator instance.
        Args:
            evaluator: Evaluator used to execute one sampling observation at a
                time for the configured task and model.
        Returns:
            Summary data containing the task identifier, model identifiers,
            number of observations, consumed tokens, configured budget,
            estimated theta, confidence interval bounds, confidence interval
            width, confidence interval method, and measurements file path.
        Raises:
            Exception: Re-raises any exception caught while creating or storing
                one observation after removing its reservation.
        """
        model_name: str = evaluator.model_name
        model_id: str = evaluator.model_id
        experiment_name: str = evaluator.experiment_name
        metric_type: str = evaluator.metric_type
        while True:
            theta_values, consumed_tokens = read_current_theta_and_total_tokens(
                self.settings,
                experiment_name,
                model_id,
            )
            if len(theta_values) >= self.settings.n_min and (consumed_tokens >= self.settings.budget_tokens
                                                        or self._ci_reached(theta_values, metric_type)):
                break
            execution_number = reserve_execution_number(
                self.settings,
                experiment_name,
                model_id,
            )
            if self.settings.inter_invocation_waiting > 0:
                time.sleep(self.settings.inter_invocation_waiting)
            try:
                observation: SamplingObservation = run_evaluator(
                    evaluator,
                    {
                        "task_id": self.settings.task_id,
                        "execution_number": execution_number,
                    },
                )
                row = create_measurement_row(
                    self.settings,
                    execution_number,
                    observation,
                    type(evaluator).__name__,
                    metric_type,
                )
                append_measurement_process_safe(
                    self.settings,
                    row,
                    experiment_name,
                    model_id,
                )
            except Exception:
                with interprocess_file_lock(self.settings.lock_path):
                    remove_reservation(
                        self.settings,
                        execution_number,
                        experiment_name,
                        model_id,
                    )
                raise

        theta_values, consumed_tokens = read_current_theta_and_total_tokens(
            self.settings,
            experiment_name,
            model_id,
        )
        ci_low, ci_high, ci_method = confidence_interval(
            theta_values,
            metric_type,
            self.settings.confidence_level,
            self.settings.ci_method,
            config.bootstrap_samples,
        )
        return {
            "task_id": self.settings.task_id,
            "model_name": model_name,
            "model_id": model_id,
            "n": len(theta_values),
            "total_tokens": consumed_tokens,
            "budget_tokens": self.settings.budget_tokens,
            "theta_hat": statistics.fmean(theta_values) if theta_values else math.nan,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width": ci_high - ci_low,
            "ci_method": ci_method,
            "measurements_path": str(self.settings.measurements_path),
        }
