"""Object-oriented example myevaluators for the adaptive sampler."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sampling.evaluators import Evaluator


class BaseEvaluator(Evaluator):
    """Shared state management for example evaluator implementations."""

    def __init__(self, **parameters: Any) -> None:
        """Initialize shared result and token counters.
        Args:
            **parameters: Evaluator-specific constructor parameters.
        """
        super().__init__(**parameters)

    @property
    def theta(self) -> float:
        """Return the numeric result produced by the last evaluator run.
        Returns:
            The most recent numeric evaluation result.
        """
        return self._theta

    @property
    def prompt_tokens(self) -> int:
        """Return the prompt token count from the last evaluator run.
        Returns:
            The most recent prompt token count.
        """
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Return the completion token count from the last evaluator run.
        Returns:
            The most recent completion token count.
        """
        return self._completion_tokens


    def _parameter(self, name: str, default: Any) -> Any:
        """Return a constructor parameter value or the provided default.
        Args:
            name: Parameter name to read.
            default: Value returned when the parameter is absent.
        Returns:
            The configured parameter value or the provided default.
        """
        return self.parameters.get(name, default)

    def _set_attribute_from_parameter(
        self,
        attribute_name: str,
        parameter_name: str,
        default: Any,
        converter: Callable[[Any], Any] | None = None,
    ) -> None:
        """Set an instance attribute from a parameter or a default value.
        Args:
            attribute_name: Instance attribute name to assign.
            parameter_name: Evaluator parameter name to read.
            default: Value assigned when the parameter is absent.
            converter: Optional callable used to convert the selected value.
        Raises:
            Exception: Re-raises any exception produced by the converter.
        """
        raw_value: Any = self._parameter(parameter_name, default)
        attribute_value: Any = converter(raw_value) if converter else raw_value
        setattr(self, attribute_name, attribute_value)

    def _load_api_key(self, provider: str, api_keys_file: str) -> str:
        """Load one provider API key from the configured JSON file.
        Args:
            provider: Provider key to read from the JSON file.
            api_keys_file: Path to the API keys JSON file.
        Returns:
            The provider API key.
        Raises:
            FileNotFoundError: If the API keys file does not exist.
            KeyError: If no API key is configured for the provider.
        """
        path: Path = Path(api_keys_file)
        if not path.exists():
            raise FileNotFoundError(f"API key file not found: {path}")
        api_keys: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        api_key: Any = api_keys.get(provider)
        if not api_key:
            raise KeyError(f"No API key configured for provider '{provider}' in {path}")
        return str(api_key)

    def _extract_numeric_quality(self, text: str) -> float:
        """Extract the first numeric quality theta from model text.
        Args:
            text: Model output text to parse.
        Returns:
            The first numeric theta found in the text.
        Raises:
            ValueError: If no numeric theta can be extracted.
        """
        match: re.Match[str] | None = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            message: str = f"Could not extract a numeric quality theta from: {text!r}"
            raise ValueError(message)
        return float(match.group(0))
