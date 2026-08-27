"""Object-oriented example myevaluators for the adaptive sampler."""

import json
import re
from pathlib import Path
from typing import Any

from sampling.evaluators import Evaluator


class BaseEvaluator(Evaluator):
    """Shared state management for example evaluator implementations."""

    def __init__(self, **parameters: Any) -> None:
        """Initialize shared result and token counters."""
        super().__init__(**parameters)
        self._theta: float = 0.0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

    @property
    def theta(self) -> float:
        """Return the numeric result produced by the last evaluator run."""
        return self._theta

    @property
    def prompt_tokens(self) -> int:
        """Return the prompt token count from the last evaluator run."""
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Return the completion token count from the last evaluator run."""
        return self._completion_tokens

    def _parameter(self, name: str, default: Any) -> Any:
        """Return a constructor parameter value or the provided default."""
        return self.parameters.get(name, default)

    def _load_api_key(self, provider: str, api_keys_file: str) -> str:
        """Load one provider API key from the configured JSON file."""

        path: Path = Path(api_keys_file)
        if not path.exists():
            raise FileNotFoundError(f"API key file not found: {path}")
        api_keys: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        api_key: Any = api_keys.get(provider)
        if not api_key:
            raise KeyError(f"No API key configured for provider '{provider}' in {path}")
        return str(api_key)

    def _extract_numeric_quality(self, text: str) -> float:
        """Extract the first numeric quality value from model text."""

        match: re.Match[str] | None = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            raise ValueError(f"Could not extract a numeric quality value from: {text!r}")
        return float(match.group(0))
