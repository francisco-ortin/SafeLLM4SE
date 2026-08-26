"""Built-in and dynamically loaded evaluator callables."""

import importlib
import json
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from sampling.config import config
from sampling.models import SamplingObservation


def coerce_observation(raw: Any) -> SamplingObservation:
    if isinstance(raw, SamplingObservation):
        return raw
    if isinstance(raw, bool):
        return SamplingObservation(value=float(int(raw)), passed=int(raw))
    if isinstance(raw, (int, float)):
        value = float(raw)
        return SamplingObservation(
            value=value,
            passed=int(value) if value in (0.0, 1.0) else None,
        )
    if isinstance(raw, dict):
        value = raw.get("value", raw.get("theta", raw.get("quality", raw.get("passed"))))
        if value is None:
            raise ValueError(
                "Evaluator dictionaries must include value, theta, quality, or passed."
            )
        passed = raw.get("passed")
        token_usage = raw.get("token_usage") or {}
        metadata = {
            key: val
            for key, val in raw.items()
            if key not in {"value", "theta", "quality", "passed", "token_usage"}
        }
        numeric_value = float(value)
        return SamplingObservation(
            value=numeric_value,
            passed=None if passed is None else int(bool(passed)),
            prompt_tokens=int(
                raw.get("prompt_tokens", token_usage.get("prompt_tokens", 0)) or 0
            ),
            completion_tokens=int(
                raw.get("completion_tokens", token_usage.get("completion_tokens", 0))
                or 0
            ),
            total_tokens=int(
                raw.get("total_tokens", token_usage.get("total_tokens", 0)) or 0
            ),
            metadata=metadata,
        )
    raise TypeError(f"Unsupported evaluator result type: {type(raw)!r}")


def gemini_quality(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **_: Any,
) -> SamplingObservation:
    """Example evaluator that calls Gemini and parses a numeric quality theta."""

    from google import genai
    from google.genai import types

    model_name = model or config.gemini_model
    client = genai.Client(api_key=_load_api_key("gemini", config.api_keys_file))
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=config.system_prompt,
            temperature=temperature if temperature is not None else config.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else config.max_tokens,
        ),
    )
    text = getattr(response, "text", "") or ""
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_token_count", 0) or prompt_tokens + completion_tokens
    )
    return SamplingObservation(
        value=_extract_numeric_quality(text),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        metadata={"provider": "gemini", "raw_text": text},
    )


def ollama_quality(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **_: Any,
) -> SamplingObservation:
    """Example evaluator that calls Ollama and parses a numeric quality theta."""

    model_name = model or config.ollama_model
    host = config.ollama_host.rstrip("/") + "/"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": temperature if temperature is not None else config.temperature,
            "num_predict": max_tokens if max_tokens is not None else config.max_tokens,
        },
        "stream": False,
    }
    request = Request(
        urljoin(host, "api/chat"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300.0) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exception:
        raise RuntimeError(
            f"Ollama request failed with HTTP {exception.code}"
        ) from exception
    except URLError as exception:
        raise RuntimeError(f"Ollama is not reachable at {host}") from exception

    text = str((response_data.get("message") or {}).get("content") or "")
    prompt_tokens = int(response_data.get("prompt_eval_count") or 0)
    completion_tokens = int(response_data.get("eval_count") or 0)
    return SamplingObservation(
        value=_extract_numeric_quality(text),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        metadata={"provider": "ollama", "raw_text": text},
    )


def random_binary(**_: Any) -> SamplingObservation:
    """Example evaluator that returns random binary outcomes."""

    passed = random.randint(0, 1)
    return SamplingObservation(value=float(passed), passed=passed)


def random_normal(**_: Any) -> SamplingObservation:
    """Example evaluator that returns real-valued quality scores in [0, 100]."""

    value = min(100.0, max(0.0, random.gauss(mu=70.0, sigma=12.0)))
    return SamplingObservation(value=value)


BUILTIN_EVALUATORS: dict[str, Callable[..., Any]] = {
    "gemini": gemini_quality,
    "gemini_quality": gemini_quality,
    "ollama": ollama_quality,
    "ollama_quality": ollama_quality,
    "random_binary": random_binary,
    "random_normal": random_normal,
}


def load_evaluator(reference: str) -> Callable[..., Any]:
    if reference in BUILTIN_EVALUATORS:
        return BUILTIN_EVALUATORS[reference]
    if ":" not in reference:
        raise ValueError(
            f"Unknown evaluator '{reference}'. Use one of {sorted(BUILTIN_EVALUATORS)} "
            "or a 'module:callable' reference."
        )
    module_name, object_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    evaluator = getattr(module, object_name)
    if not callable(evaluator):
        raise TypeError(f"{reference} is not callable.")
    return evaluator


def _load_api_key(provider: str, api_keys_file: str) -> str:
    path = Path(api_keys_file)
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    api_keys = json.loads(path.read_text(encoding="utf-8"))
    api_key = api_keys.get(provider)
    if not api_key:
        raise KeyError(f"No API key configured for provider '{provider}' in {path}")
    return str(api_key)


def _extract_numeric_quality(text: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"Could not extract a numeric quality value from: {text!r}")
    return float(match.group(0))
