"""API key loading helpers for configured LLM providers."""

import json


def read_api_key_for_model(file_path: str, model_id: str) -> str:
    """Read the API key assigned to a model id from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        api_keys: dict[str, str] = json.load(file)

    api_key: str | None = api_keys.get(model_id)
    if not api_key:
        raise KeyError(
            f"No API key configured for model id '{model_id}' in {file_path}."
        )
    return api_key
