"""Ollama client implementation for local LLM completions."""

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from llms.base import LLMCompletion


DEFAULT_OLLAMA_HOST: str = "http://127.0.0.1:11434"
OLLAMA_CHAT_PATH: str = "/api/chat"


def _normalize_ollama_host(host: str) -> str:
    """Return a normalized Ollama host URL with a trailing slash."""
    stripped_host: str = host.strip()
    if not stripped_host:
        stripped_host = DEFAULT_OLLAMA_HOST
    if not stripped_host.startswith(("http://", "https://")):
        stripped_host = f"http://{stripped_host}"
    return stripped_host.rstrip("/") + "/"


def _extract_ollama_text(response: dict[str, Any]) -> str:
    """Extract generated text from an Ollama chat response."""
    message: dict[str, Any] = response.get("message") or {}
    content: Any = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def _extract_ollama_token_usage(response: dict[str, Any]) -> dict[str, int]:
    """Extract Ollama token usage into the common completion schema."""
    prompt_tokens: int = int(response.get("prompt_eval_count") or 0)
    completion_tokens: int = int(response.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class OllamaLLMClient:
    """Client wrapper for Ollama local chat completions."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        host: str | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        """Initialize the Ollama HTTP client for a concrete model."""
        self.model_name = model_name
        self.api_key = api_key
        self.host = _normalize_ollama_host(
            host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        )
        self.timeout_seconds = timeout_seconds

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        """Generate one completion and return text with token usage."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        request: Request = Request(
            url=urljoin(self.host, OLLAMA_CHAT_PATH.lstrip("/")),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_data: dict[str, Any] = json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exception:
            error_message: str = exception.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request failed with HTTP {exception.code}: {error_message}"
            ) from exception
        except URLError as exception:
            raise RuntimeError(
                "Ollama is not reachable. Check that the Ollama service is "
                f"running at {self.host}. If this code runs inside Docker and "
                "Ollama runs on the host machine, set OLLAMA_HOST to "
                "http://host.docker.internal:11434 or use host networking."
            ) from exception

        return LLMCompletion(
            text=_extract_ollama_text(response_data),
            token_usage=_extract_ollama_token_usage(response_data),
        )
