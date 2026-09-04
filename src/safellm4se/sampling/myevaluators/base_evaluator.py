"""Object-oriented example myevaluators for the adaptive sampler."""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from safellm4se.sampling.evaluators import Evaluator


PARAMETER_ALIASES: dict[str, tuple[str, ...]] = {
    "standard_deviation": ("standard-deviation",),
    "success_probability": ("success-probability",),
}  # Alternative evaluator parameter names accepted by shared evaluators.

DEFAULT_ENV_FILE: str = ".env"  # Default dotenv file read when variables are unset.


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
        if name in self.parameters:
            return self.parameters[name]
        for alias in PARAMETER_ALIASES.get(name, ()):
            if alias in self.parameters:
                return self.parameters[alias]
        return default

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

    def _load_api_key_from_environment(
        self,
        environment_variable_name: str,
        env_file: str = DEFAULT_ENV_FILE,
    ) -> str:
        """Load one API key from the process environment or a dotenv file.
        Args:
            environment_variable_name: Environment variable that stores the API key.
            env_file: Dotenv file used as fallback when the process variable is unset.
        Returns:
            The configured API key.
        Raises:
            KeyError: If the API key is absent from both sources.
        """
        return self._load_environment_value(
            environment_variable_name=environment_variable_name,
            value_description="API key",
            env_file=env_file,
        )

    def _load_environment_value(
        self,
        environment_variable_name: str,
        value_description: str,
        env_file: str = DEFAULT_ENV_FILE,
    ) -> str:
        """Load one configuration value from the environment or a dotenv file.
        Args:
            environment_variable_name: Environment variable that stores the value.
            value_description: Human-readable value description for errors.
            env_file: Dotenv file used as fallback when the process variable is unset.
        Returns:
            The configured environment value.
        Raises:
            KeyError: If the value is absent from both sources.
        """
        environment_value: str | None = os.environ.get(environment_variable_name)
        if environment_value:
            return environment_value

        dotenv_values: dict[str, str] = _read_dotenv_values(Path(env_file))
        environment_value = dotenv_values.get(environment_variable_name)
        if not environment_value:
            raise KeyError(
                f"No {value_description} configured in "
                f"{environment_variable_name} or {Path(env_file)}."
            )
        return environment_value

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


def _read_dotenv_values(path: Path) -> dict[str, str]:
    """Read dotenv key-value pairs from a file when it exists.
    Args:
        path: Dotenv file path.
    Returns:
        Parsed environment values keyed by variable name.
    """
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line: str = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        raw_name: str
        raw_value: str
        raw_name, raw_value = line.split("=", 1)
        name: str = raw_name.strip()
        value: str = _clean_dotenv_value(raw_value.strip())
        if name:
            values[name] = value
    return values


def _clean_dotenv_value(raw_value: str) -> str:
    """Remove optional dotenv quotes and inline comments from a value.
    Args:
        raw_value: Raw dotenv value text.
    Returns:
        Cleaned dotenv value.
    """
    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {
        "'",
        '"',
    }:
        return raw_value[1:-1]
    value, _separator, _comment = raw_value.partition(" #")
    return value.strip()
