"""Base data structures and protocols shared by all LLM clients."""

from dataclasses import dataclass
from typing import Protocol


class LLMRateLimitError(Exception):
    """Raised when the selected LLM provider reports a rate limit."""


class LLMQuotaExceededError(LLMRateLimitError):
    """Raised when the selected LLM provider reports exhausted account quota."""


@dataclass(frozen=True)
class LLMCompletion:
    """Generated text and token usage returned by an LLM provider."""

    text: str
    token_usage: dict[str, int]


class LLMClient(Protocol):
    """Protocol implemented by concrete LLM provider clients."""

    model_name: str

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        """Generate one completion for a system and user prompt."""
        ...
