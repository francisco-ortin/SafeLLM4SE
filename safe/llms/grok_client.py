"""Grok client implementation for LLM completions."""

from typing import Any

from llms.base import LLMCompletion, LLMRateLimitError


def _usage_value(usage: Any, field_name: str) -> int:
    """Read an integer usage field from SDK objects or dictionaries."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(field_name) or 0)
    return int(getattr(usage, field_name, 0) or 0)


def _extract_grok_token_usage(response: Any) -> dict[str, int]:
    """Extract Grok token usage into the common completion schema."""
    usage: Any = getattr(response, "usage", None)
    prompt_tokens: int = _usage_value(usage, "prompt_tokens")
    completion_tokens: int = _usage_value(usage, "completion_tokens")
    total_tokens: int = (
        _usage_value(usage, "total_tokens")
        or prompt_tokens + completion_tokens
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


class GrokLLMClient:
    """Client wrapper for Grok chat completions."""

    def __init__(self, api_key: str, model_name: str) -> None:
        """Initialize the Grok SDK client for a concrete model."""
        from groq import Groq

        self.model_name = model_name
        self._client = Groq(api_key=api_key)

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        """Generate one completion and return text with token usage."""
        from groq import RateLimitError

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as exception:
            raise LLMRateLimitError(str(exception)) from exception

        return LLMCompletion(
            text=response.choices[0].message.content or "",
            token_usage=_extract_grok_token_usage(response),
        )
