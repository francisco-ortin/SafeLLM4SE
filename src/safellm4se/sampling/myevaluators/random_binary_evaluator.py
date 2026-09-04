"""Example random binary evaluator for the adaptive sampler."""

import random
from typing import Any

from safellm4se.sampling.myevaluators.base_evaluator import BaseEvaluator
from safellm4se.sampling.models import SamplingObservation

EXPERIMENT_NAME: str = "random-binary"
MODEL_NAME: str = "random-binary"
MODEL_ID: str = "random-binary-v0"
DEFAULT_SUCCESS_PROBABILITY: float = 0.5


class RandomBinaryEvaluator(BaseEvaluator):
    """Example evaluator that returns random binary outcomes."""

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The random binary model name.
        """
        return str(self._parameter("model-name", MODEL_NAME))

    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The random binary experiment name.
        """
        return str(self._parameter("experiment-name", EXPERIMENT_NAME))

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The configured random binary model identifier.
        """
        return str(self._parameter("model_id", MODEL_ID))

    def __init__(
        self,
        success_probability: float = DEFAULT_SUCCESS_PROBABILITY,
        **parameters: Any,
    ) -> None:
        """Initialize the evaluator with a configurable success probability.
        Args:
            success_probability: Probability of generating a successful outcome.
            **parameters: Evaluator parameters, including required temperature.
        Raises:
            ValueError: If success_probability is outside the [0, 1] range.
            TypeError: If success_probability cannot be converted to float.
        """
        super().__init__(**parameters)
        raw_success_probability: Any = self._parameter(
            "success_probability",
            success_probability,
        )
        # Probability that a generated random value is converted into success.
        self.success_probability: float = _validate_success_probability(
            raw_success_probability,
        )

    @property
    def metric_type(self) -> str:
        """Return the binary variable type used by this evaluator.
        Returns:
            The binary metric type.
        """
        return "binary"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Generate one random binary observation and update evaluator state.
        Args:
            **context: Unused runtime context values.
        Returns:
            A sampling observation containing the random binary outcome and token
            counts.
        """
        del context
        self._theta = float(random.random() < self.success_probability)
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


def _validate_success_probability(value: Any) -> float:
    """Return a validated success probability.
    Args:
        value: Candidate success probability.
    Returns:
        The success probability as a float.
    Raises:
        ValueError: If value is outside the [0, 1] range.
        TypeError: If value cannot be converted to float.
    """
    success_probability: float = float(value)
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must be between 0 and 1.")
    return success_probability

