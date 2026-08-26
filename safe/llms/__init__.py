"""Factory and public API for supported LLM provider clients."""

from typing import TypeAlias

from sampling.config.config import Config
from llms.api_keys import read_api_key_for_model
from llms.base import (
    LLMClient,
    LLMCompletion,
    LLMQuotaExceededError,
    LLMRateLimitError,
)
from llms.gemini_client import GeminiLLMClient
from llms.grok_client import GrokLLMClient
from llms.ollama_client import OllamaLLMClient

LLMClientClass: TypeAlias = (
        type[GrokLLMClient] | type[GeminiLLMClient] | type[OllamaLLMClient]
)

_CLIENTS_BY_PROVIDER: dict[str, LLMClientClass] = {
    "grok": GrokLLMClient,
    "gemini": GeminiLLMClient,
    "ollama": OllamaLLMClient,
}

_PROVIDERS_REQUIRING_API_KEYS: set[str] = {"grok", "gemini"}


def create_llm_client(model_id: str, model_name: str, api_keys_path: str) -> LLMClient:
    """Create an LLM client for the configured model id."""
    provider_name: str = Config.get_model_provider(model_id)
    try:
        client_class: LLMClientClass = _CLIENTS_BY_PROVIDER[provider_name]
    except KeyError as exception:
        raise ValueError(
            f"No LLM client configured for provider '{provider_name}'."
        ) from exception

    api_key: str | None = None
    if provider_name in _PROVIDERS_REQUIRING_API_KEYS:
        api_key = read_api_key_for_model(api_keys_path, model_id)
    if provider_name == "ollama":
        return OllamaLLMClient(model_name=model_name, host=Config.OLLAMA_HOST)
    return client_class(api_key=api_key, model_name=model_name)


__all__ = [
    "LLMClient",
    "LLMCompletion",
    "LLMQuotaExceededError",
    "LLMRateLimitError",
    "OllamaLLMClient",
    "create_llm_client",
]
