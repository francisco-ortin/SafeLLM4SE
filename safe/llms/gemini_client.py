"""Google Gemini client implementation for LLM completions."""

from typing import Any

from llms.base import LLMCompletion, LLMQuotaExceededError, LLMRateLimitError


def _usage_value(usage: Any, field_name: str) -> int:
    """Read an integer usage field from SDK objects or dictionaries."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(field_name) or 0)
    return int(getattr(usage, field_name, 0) or 0)


def _extract_gemini_token_usage(response: Any) -> dict[str, int]:
    """Extract Gemini token usage into the common completion schema."""
    usage: Any = getattr(response, "usage_metadata", None)
    prompt_tokens: int = _usage_value(usage, "prompt_token_count")
    completion_tokens: int = _usage_value(usage, "candidates_token_count")
    total_tokens: int = (
        _usage_value(usage, "total_token_count")
        or prompt_tokens + completion_tokens
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_gemini_text(response: Any) -> str:
    """Extract text from Gemini responses across SDK response shapes."""
    try:
        text: str | None = getattr(response, "text", None)
    except Exception:
        text = None
    if text:
        return text

    candidates: list[Any] = getattr(response, "candidates", None) or []
    parts_text: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts_text.append(part_text)
    return "\n".join(parts_text)


def _is_rate_limit_error(exception: Exception) -> bool:
    """Return whether an exception represents a provider rate-limit error."""
    status_code: int | None = (
        getattr(exception, "code", None)
        or getattr(exception, "status_code", None)
    )
    if status_code == 429:
        return True
    message = str(exception).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "rate limit" in message
    )


def _is_quota_exhausted_error(exception: Exception) -> bool:
    """Return whether an exception indicates an exhausted project quota."""
    message: str = str(exception).lower()
    return (
        "quota exceeded" in message
        or "limit: 0" in message
        or "billing details" in message
    )


class GeminiLLMClient:
    """Client wrapper for Google Gemini text generation."""

    def __init__(self, api_key: str, model_name: str) -> None:
        """Initialize the Gemini SDK client for a concrete model."""
        from google import genai

        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        """Generate one completion and return text with token usage."""
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
        except Exception as exception:
            if _is_rate_limit_error(exception):
                if _is_quota_exhausted_error(exception):
                    raise LLMQuotaExceededError(str(exception)) from exception
                raise LLMRateLimitError(str(exception)) from exception
            raise

        return LLMCompletion(
            text=_extract_gemini_text(response),
            token_usage=_extract_gemini_token_usage(response),
        )
