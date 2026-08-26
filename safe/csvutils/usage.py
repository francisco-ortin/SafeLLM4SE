"""Token usage extraction helpers for LLM provider responses."""

from typing import Any


def _usage_value(usage: Any, field_name: str) -> int:
    """Reads token usage from either SDK objects or dict-like responses."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(field_name) or 0)
    return int(getattr(usage, field_name, 0) or 0)


def extract_token_usage(response: Any) -> dict[str, int]:
    """Extract prompt, completion, and total token counts from a response."""
    usage: Any = getattr(response, "usage", None)
    prompt_tokens: int = _usage_value(usage, "prompt_tokens")
    completion_tokens: int = _usage_value(usage, "completion_tokens")
    total_tokens: int = _usage_value(usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
