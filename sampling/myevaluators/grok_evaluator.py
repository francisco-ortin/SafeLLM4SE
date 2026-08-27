"""Grok API evaluator for the adaptive sampler."""

from typing import Any

from sampling.config import config
from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation


class GrokQualityEvaluator(BaseEvaluator):
    """Evaluator that calls Grok and parses a numeric quality score."""

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model and temperature settings."""

        super().__init__(**parameters)
        assert (
            "temperature" in parameters
        ), "This model requires temperature parameter as a float"
        assert isinstance(parameters["temperature"], (int, float)), (
            "The temperature parameter must be a float or integer"
        )
        self.temperature: float = float(parameters["temperature"])
        if "model_id" in parameters:
            self._model_id: str = str(parameters["model_id"])
        elif "model" in parameters:
            self._model_id = str(parameters["model"])
        else:
            self._model_id = "grok-4.1-fast"

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements."""
        return "grok"

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider."""
        return self._model_id

    @property
    def metric_type(self) -> str:
        """Return the continuous variable type used by this evaluator."""
        return "continuous"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Call Grok, parse the result, and update evaluator state."""

        from groq import Groq, RateLimitError

        prompt: str = str(self._parameter("prompt", context.get("prompt", "")))
        model_id: str = str(self._parameter("model", self.model_id))
        temperature: float = float(
            self._parameter(
                "temperature",
                context.get("temperature", self.temperature),
            )
        )
        max_tokens: int = int(
            self._parameter("max_tokens", context.get("max_tokens", config.max_tokens))
        )
        api_keys_file: str = str(self._parameter("api_keys_file", config.api_keys_file))
        system_prompt: str = str(self._parameter("system_prompt", config.system_prompt))

        client = Groq(api_key=self._load_api_key("grok", api_keys_file))
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as exception:
            raise RuntimeError(
                f"Grok request failed due to rate limiting: {exception}"
            ) from exception

        text: str = response.choices[0].message.content or ""
        usage: Any = getattr(response, "usage", None)
        self._prompt_tokens = _usage_value(usage, "prompt_tokens")
        self._completion_tokens = _usage_value(usage, "completion_tokens")
        self._theta = self._extract_numeric_quality(text)
        return SamplingObservation(
            theta=self._theta,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            metadata={"provider": "grok", "raw_text": text},
        )


def _usage_value(usage: Any, field_name: str) -> int:
    """Read an integer usage field from SDK objects or dictionaries."""

    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(field_name) or 0)
    return int(getattr(usage, field_name, 0) or 0)

