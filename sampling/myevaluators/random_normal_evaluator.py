"""Example random continuous normal evaluator for the adaptive sampler."""

import random
from typing import Any

from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation


class RandomNormalEvaluator(BaseEvaluator):
    """Example evaluator that returns real-valued quality scores in [0, 100]."""

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements."""
        return "random-normal"

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider."""
        return str(self._parameter("model_id", "random-normal-v0"))

    def __init__(
        self,
        mean: float = 70.0,
        standard_deviation: float = 12.0,
        **parameters: Any,
    ) -> None:
        """Initialize the evaluator with normal distribution parameters."""
        super().__init__(**parameters)
        self.mean: float = mean
        self.standard_deviation: float = standard_deviation

    @property
    def metric_type(self) -> str:
        """Return the continuous variable type used by this evaluator."""
        return "continuous"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Generate one random continuous observation and update evaluator state."""
        del context
        self._theta = min(
            100.0,
            max(0.0, random.gauss(mu=self.mean, sigma=self.standard_deviation)),
        )
        self._prompt_tokens = random.randint(100, 1000)
        self._completion_tokens = random.randint(100, 1000)
        return SamplingObservation(
            theta=self._theta,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._completion_tokens + self._prompt_tokens,
        )
