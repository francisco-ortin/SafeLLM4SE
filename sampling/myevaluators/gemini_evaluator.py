"""Gemini API evaluator for the adaptive sampler."""

from typing import Any

from sampling.config import config
from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation


class GeminiQualityEvaluator(BaseEvaluator):
    """Example evaluator that calls Gemini and parses a numeric quality score."""

    @property
    def metric_type(self) -> str:
        """Return the continuous variable type used by this evaluator."""

        return "continuous"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Call Gemini, parse the result, and update evaluator state."""

        from google import genai
        from google.genai import types

        prompt: str = str(self._parameter("prompt", context.get("prompt", "")))
        model_name: str = str(
            self._parameter("model", context.get("model", config.gemini_model))
        )
        temperature: float = float(
            self._parameter(
                "temperature",
                context.get("temperature", config.temperature),
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
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._prompt_tokens + self._completion_tokens,
            metadata={"provider": "gemini", "raw_text": text},
        )
