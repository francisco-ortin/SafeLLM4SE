"""Ollama API evaluator for the adaptive sampler."""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from sampling.config import config
from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation


class OllamaQualityEvaluator(BaseEvaluator):
    """Example evaluator that calls Ollama and parses a numeric quality score."""

    @property
    def metric_type(self) -> str:
        """Return the continuous variable type used by this evaluator."""

        return "continuous"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Call Ollama, parse the result, and update evaluator state."""

        prompt: str = str(self._parameter("prompt", context.get("prompt", "")))
        model_name: str = str(
            self._parameter("model", context.get("model", config.ollama_model))
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
        host: str = str(self._parameter("ollama_host", config.ollama_host))
        system_prompt: str = str(self._parameter("system_prompt", config.system_prompt))
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        request = Request(
            urljoin(host.rstrip("/") + "/", "api/chat"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=300.0) as response:
                response_data: dict[str, Any] = json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exception:
            raise RuntimeError(
                f"Ollama request failed with HTTP {exception.code}"
            ) from exception
        except URLError as exception:
            raise RuntimeError(f"Ollama is not reachable at {host}") from exception

        text: str = str((response_data.get("message") or {}).get("content") or "")
        self._prompt_tokens = int(response_data.get("prompt_eval_count") or 0)
        self._completion_tokens = int(response_data.get("eval_count") or 0)
        self._theta = self._extract_numeric_quality(text)
        return SamplingObservation(
            theta=self._theta,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._prompt_tokens + self._completion_tokens,
            metadata={"provider": "ollama", "raw_text": text},
        )
