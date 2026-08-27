"""Ollama API evaluator for the adaptive sampler."""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from sampling.config import config
from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation

# Values of parameters if not passed in the command line
MODEL_ID: str = "qwen2.5-coder:7b"  # Unique universal model identifier.
MODEL_NAME: str = "qwen-coder"  # Short model name.
# Name of the experiment represented by this evaluator.
EXPERIMENT_NAME: str = "ollama-random"
MAX_TOKENS: int = 256  # Maximum number of tokens for the LLM response.
OLLAMA_HOST: str = "http://host.docker.internal:11434"  # Host API for Ollama.
PROMPT: str = "Give me a random float value between 0 and 1"


class OllamaRandomEvaluator(BaseEvaluator):
    """Example evaluator that calls Ollama and parses a numeric quality score."""

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model and temperature settings.
        Args:
            **parameters: Evaluator parameters, including required temperature
                and optional model or model_id.
        Raises:
            AssertionError: If temperature is missing or is not numeric.
        """
        super().__init__(**parameters)
        assert (
            "temperature" in parameters
        ), "This model requires temperature parameter as a float"
        assert isinstance(parameters["temperature"], (int, float)), (
            "The temperature parameter must be a float or integer"
        )
        self._set_attribute_from_parameter(
            "temperature",
            "temperature",
            parameters["temperature"],
            float,
        )
        self._set_attribute_from_parameter("_model_id", "model_id", MODEL_ID, str)
        self._set_attribute_from_parameter("_model_name", "model_name", MODEL_NAME, str)
        self._set_attribute_from_parameter("_max_tokens", "max_tokens", MAX_TOKENS, int)
        self._set_attribute_from_parameter(
            "_ollama_host",
            "ollama_host",
            OLLAMA_HOST,
            str,
        )
        self._set_attribute_from_parameter("_prompt", "prompt", PROMPT, str)

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The canonical Ollama-backed model name.
        """
        return self._model_name

    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The Ollama random experiment name.
        """
        return EXPERIMENT_NAME

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The configured Ollama model identifier.
        """
        return self._model_id

    @property
    def metric_type(self) -> str:
        """Return the continuous variable type used by this evaluator.
        Returns:
            The continuous metric type.
        """
        return "continuous"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Call Ollama, parse the result, and update evaluator state.
        Args:
            **context: Runtime context values that may override prompt,
                temperature, or max token settings.
        Returns:
            A sampling observation containing the parsed quality score, model
            identifiers, token counts, and provider metadata.
        Raises:
            RuntimeError: If Ollama returns an HTTP error or cannot be reached.
        """
        system_prompt: str = str(self._parameter("system_prompt", config.system_prompt))
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": int(self._max_tokens),
            },
            "stream": False,
        }
        request = Request(
            urljoin(self._ollama_host.rstrip("/") + "/", "api/chat"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5000.0) as response:
                response_data: dict[str, Any] = json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exception:
            raise RuntimeError(
                f"Ollama request failed with HTTP {exception.code}"
            ) from exception
        except URLError as exception:
            raise RuntimeError(
                f"Ollama is not reachable at {self._ollama_host}"
            ) from exception

        text: str = str((response_data.get("message") or {}).get("content") or "")
        self._prompt_tokens = int(response_data.get("prompt_eval_count") or 0)
        self._completion_tokens = int(response_data.get("eval_count") or 0)
        self._theta = self._extract_numeric_quality(text)

        return SamplingObservation(
            theta=self._theta,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            metadata={"provider": "ollama", "raw_text": text},
        )
