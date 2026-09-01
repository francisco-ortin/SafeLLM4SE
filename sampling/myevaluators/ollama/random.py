"""Ollama random numeric evaluator for the adaptive sampler."""

from typing import Any

from sampling.config import config
from sampling.myevaluators.ollama.common import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_HOST,
    OllamaBaseEvaluator,
    response_completion_tokens,
    response_prompt_tokens,
    response_text,
)
from sampling.models import SamplingObservation

MODEL_ID: str = DEFAULT_MODEL_ID  # Unique universal model identifier.
MODEL_NAME: str = DEFAULT_MODEL_NAME  # Short model name.
EXPERIMENT_NAME: str = "ollama-random"  # Experiment represented by this evaluator.
MAX_TOKENS: int = 256  # Maximum number of tokens for the LLM response.
OLLAMA_HOST: str = DEFAULT_OLLAMA_HOST  # Host API for Ollama.
PROMPT: str = "Give me a random float value between 0 and 1"


class OllamaRandomEvaluator(OllamaBaseEvaluator):
    """Evaluator that calls Ollama and parses a numeric quality score."""

    EXPERIMENT_NAME: str = EXPERIMENT_NAME
    METRIC_TYPE: str = "continuous"

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model and prompt settings.
        Args:
            **parameters: Evaluator parameters. Defaults are provided if not specified.
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
        """Call Ollama, parse the result, and update evaluator state.
        Args:
            **context: Runtime context values. They are ignored by this evaluator.
        Returns:
            A sampling observation containing the parsed quality score, model
            identifiers, token counts, and provider metadata.
        Raises:
            RuntimeError: If Ollama returns an HTTP error or cannot be reached.
            ValueError: If the Ollama response does not contain a numeric value.
        """
        del context
        response_data: dict[str, Any] = self.call_ollama(self._prompt)
        text: str = response_text(response_data)
        theta: float = self._extract_numeric_quality(text)

        return self.build_observation(
            theta=theta,
            prompt_tokens=response_prompt_tokens(response_data),
            completion_tokens=response_completion_tokens(response_data),
            metadata={"provider": "ollama", "raw_text": text},
        )
