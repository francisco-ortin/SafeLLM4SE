"""Gemini API evaluator for the adaptive sampler."""

from typing import Any

from sampling.config import config
from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation


class GeminiQualityEvaluator(BaseEvaluator):
    """Example evaluator that calls Gemini and parses a numeric quality score."""

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with a configurable success probability."""
        super().__init__(**parameters)
        assert (
            "temperature" in parameters
        ), "This model requires temperature parameter as a float"
        assert isinstance(parameters["temperature"], (int, float)), (
            "The temperature parameter must be a float or integer"
        )
        self.temperature: float = parameters["temperature"]
        if "model_id" in parameters:
            self._model_id: str = str(parameters["model_id"])
        elif "model" in parameters:
            self._model_id = str(parameters["model"])
        else:
            self._model_id: str = "gemini-3.1-flash-lite"

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements."""
        return "gemini"

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider."""
        return self._model_id

    @property
    def metric_type(self) -> str:
        """Return the continuous variable type used by this evaluator."""
        return "continuous"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Call Gemini, parse the result, and update evaluator state."""

        from google import genai
        from google.genai import types

        prompt: str = str(self._parameter("prompt", context.get("prompt", "")))
        model_name: str = str(self._parameter("model", self.model_id))
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

        client = genai.Client(api_key=self._load_api_key("gemini", api_keys_file))
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        text: str = getattr(response, "text", "") or ""
        usage: Any = getattr(response, "usage_metadata", None)
        self._prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        self._completion_tokens = int(
            getattr(usage, "candidates_token_count", 0) or 0
        )
        self._theta = self._extract_numeric_quality(text)
        return SamplingObservation(
            theta=self._theta,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._prompt_tokens + self._completion_tokens,
            metadata={"provider": "gemini", "raw_text": text},
        )
