"""Gemini random numeric evaluator for the adaptive sampler."""

from typing import Any

from sampling.config import config
from sampling.myevaluators.gemini.common import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    GeminiBaseEvaluator,
    response_completion_tokens,
    response_prompt_tokens,
    response_text,
)
from sampling.models import SamplingObservation

MODEL_ID: str = DEFAULT_MODEL_ID  # Unique universal model identifier.
MODEL_NAME: str = DEFAULT_MODEL_NAME  # Short model name.
EXPERIMENT_NAME: str = "gemini-random"  # Experiment represented by this evaluator.
MAX_TOKENS: int = 256  # Maximum number of tokens for the LLM response.
PROMPT: str = "Give me a random float value between 0 and 1"


class GeminiRandomEvaluator(GeminiBaseEvaluator):
    """Evaluator that calls Gemini and parses a numeric quality score."""

    EXPERIMENT_NAME: str = EXPERIMENT_NAME  # Experiment represented by the class.
    METRIC_TYPE: str = "continuous"  # Statistical metric type.

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model and prompt settings.
        Args:
            **parameters: Evaluator parameters. Defaults are provided if not specified.
        Returns:
            None.
        Raises:
            Exception: Re-raises any exception produced by parameter conversion.
        """
        super().__init__(
            default_max_tokens=MAX_TOKENS,
            default_system_prompt=str(config.system_prompt),
            **parameters,
        )
        self._set_attribute_from_parameter(
            "_prompt",
            "prompt",
            PROMPT,
            str,
        )  # User prompt used to request a random numeric value.

    def run(self, **context: Any) -> SamplingObservation | None:
        """Call Gemini, parse the result, and update evaluator state.
        Args:
            **context: Runtime context values. They are ignored by this evaluator.
        Returns:
            A sampling observation containing the parsed quality score, model
            identifiers, token counts, and provider metadata.
        Raises:
            RuntimeError: If the Gemini request fails.
            ValueError: If the Gemini response does not contain a numeric value.
            FileNotFoundError: If the API key file does not exist.
            KeyError: If no Gemini API key is configured.
        """
        del context
        response_data: dict[str, Any] = self.call_gemini(self._prompt)
        text: str = response_text(response_data)
        theta: float = self._extract_numeric_quality(text)

        return self.build_observation(
            theta=theta,
            prompt_tokens=response_prompt_tokens(response_data),
            completion_tokens=response_completion_tokens(response_data),
            metadata={"provider": "gemini", "raw_text": text},
        )
