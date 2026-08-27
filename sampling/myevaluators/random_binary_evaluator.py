"""Example random binary evaluator for the adaptive sampler."""

import random
from typing import Any

from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation

# Name of the experiment represented by this evaluator.
EXPERIMENT_NAME: str = "random-binary"


class RandomBinaryEvaluator(BaseEvaluator):
    """Example evaluator that returns random binary outcomes."""

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The random binary model name.
        """
        return "random-binary"

    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The random binary experiment name.
        """
        return EXPERIMENT_NAME

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The configured random binary model identifier.
        """
        return str(self._parameter("model_id", "random-binary-v0"))

    def __init__(self, success_probability: float = 0.5, **parameters: Any) -> None:
        """Initialize the evaluator with a configurable success probability.
        Args:
            success_probability: Probability of generating a successful outcome.
            **parameters: Evaluator parameters, including required temperature.
        Raises:
            AssertionError: If temperature is missing or is not numeric.
        """
        super().__init__(**parameters)
        self.success_probability: float = success_probability
        assert (
            "temperature" in parameters
        ), "This model requires temperature parameter as a float"
        assert isinstance(parameters["temperature"], (int, float)), (
            "The temperature parameter must be a float or integer"
        )
        self.temperature: float = parameters["temperature"]

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
        self._prompt_tokens = random.randint(100, 1000)
        self._completion_tokens = random.randint(100, 1000)
        return SamplingObservation(
            theta=self._theta,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._completion_tokens + self._prompt_tokens,
        )
