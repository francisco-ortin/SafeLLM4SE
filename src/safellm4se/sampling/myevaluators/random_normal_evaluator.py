"""Example random continuous normal evaluator for the adaptive sampler."""

import random
from typing import Any

from safellm4se.sampling.myevaluators.base_evaluator import BaseEvaluator
from safellm4se.sampling.models import SamplingObservation

EXPERIMENT_NAME: str = "random-normal"
MODEL_NAME: str = "random-normal"
MODEL_ID: str = "random-normal-v0"
DEFAULT_MEAN: float = 50.0
DEFAULT_STANDARD_DEVIATION: float = 25.0


class RandomNormalEvaluator(BaseEvaluator):
    """Example evaluator that returns real-valued quality scores in [0, 100]."""

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The random normal model name.
        """
        return str(self._parameter("model-name", MODEL_NAME))

    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The random normal experiment name.
        """
        return str(self._parameter("experiment-name", EXPERIMENT_NAME))

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The configured random normal model identifier.
        """
        return str(self._parameter("model_id", MODEL_ID))

    def __init__(
        self,
        mean: float = DEFAULT_MEAN,
        standard_deviation: float = DEFAULT_STANDARD_DEVIATION,
        **parameters: Any,
    ) -> None:
        """Initialize the evaluator with normal distribution parameters.
        Args:
            mean: Mean theta used by the normal distribution.
            standard_deviation: Standard deviation used by the normal
                distribution.
            **parameters: Additional evaluator parameters.
        Raises:
            ValueError: If standard_deviation is negative.
            TypeError: If mean or standard_deviation cannot be converted to float.
        """
        super().__init__(**parameters)
        raw_mean: Any = self._parameter("mean", mean)
        raw_standard_deviation: Any = self._parameter(
            "standard_deviation",
            standard_deviation,
        )
        # Mean used to center the generated normal distribution.
        self.mean: float = float(raw_mean)
        # Standard deviation used to spread the generated normal distribution.
        self.standard_deviation: float = _validate_standard_deviation(
            raw_standard_deviation,
        )

    @property
    def metric_type(self) -> str:
        """Return the continuous variable type used by this evaluator.
        Returns:
            The continuous metric type.
        """
        return "continuous"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Generate one random continuous observation and update evaluator state.
        Args:
            **context: Unused runtime context values.
        Returns:
            A sampling observation containing the random continuous theta and
            token counts.
        """
        del context
        self._theta = min(
            100.0,
            max(0.0, random.gauss(mu=self.mean, sigma=self.standard_deviation)),
        )
        self._prompt_tokens = random.randint(10, 100)
        self._completion_tokens = random.randint(10, 100)
        return SamplingObservation(
            theta=self._theta,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._completion_tokens + self._prompt_tokens,
        )


def _validate_standard_deviation(value: Any) -> float:
    """Return a validated standard deviation.
    Args:
        value: Candidate standard deviation.
    Returns:
        The standard deviation as a float.
    Raises:
        ValueError: If value is negative.
        TypeError: If value cannot be converted to float.
    """
    standard_deviation: float = float(value)
    if standard_deviation < 0.0:
        raise ValueError("standard_deviation cannot be negative.")
    return standard_deviation

