"""Evaluator contract and loading utilities for the sampling framework."""

import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

from sampling.models import SamplingObservation


class Evaluator(ABC):
    """Abstract contract implemented by sampling myevaluators."""

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with user-provided parameters."""

        self.parameters: dict[str, Any] = parameters

    @abstractmethod
    def run(self, **context: Any) -> SamplingObservation | None:
        """Execute the model call and evaluation for one sampling observation."""

    @property
    @abstractmethod
    def theta(self) -> float:
        """Return the numeric evaluation result for the last run."""

    @property
    @abstractmethod
    def metric_type(self) -> str:
        """Return the evaluated variable type: binary or continuous."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider."""

    @property
    @abstractmethod
    def prompt_tokens(self) -> int:
        """Return the number of prompt tokens consumed by the last run."""

    @property
    @abstractmethod
    def completion_tokens(self) -> int:
        """Return the number of completion tokens produced by the last run."""


def load_evaluator(
    reference: str,
    parameters: dict[str, Any] | None = None,
) -> Evaluator:
    """Load and instantiate an evaluator class from a module reference."""

    evaluator_class: type[Evaluator] = _load_evaluator_class(reference)
    return evaluator_class(**(parameters or {}))


def run_evaluator(
    evaluator: Evaluator,
    context: dict[str, Any],
) -> SamplingObservation:
    """Run an evaluator and return its result as a sampling observation."""

    result: SamplingObservation | None = evaluator.run(**context)
    if result is not None:
        observation: SamplingObservation = coerce_observation(result)
    else:
        observation = observation_from_evaluator(evaluator)
    if observation.model_name and observation.model_id:
        return observation
    return replace(
        observation,
        model_name=observation.model_name or evaluator.model_name,
        model_id=observation.model_id or evaluator.model_id,
    )


def observation_from_evaluator(evaluator: Evaluator) -> SamplingObservation:
    """Build a sampling observation from an evaluator object's public state."""

    prompt_tokens: int = evaluator.prompt_tokens
    completion_tokens: int = evaluator.completion_tokens
    theta: float = float(evaluator.theta)
    return SamplingObservation(
        theta=theta,
        model_name=evaluator.model_name,
        model_id=evaluator.model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        metadata={"evaluator_type": evaluator.metric_type},
    )


def coerce_observation(raw: Any) -> SamplingObservation:
    """Convert supported evaluator return values into a sampling observation."""

    if isinstance(raw, SamplingObservation):
        return raw
    if isinstance(raw, bool):
        return SamplingObservation(theta=float(int(raw)))
    if isinstance(raw, (int, float)):
        return SamplingObservation(theta=float(raw))
    if isinstance(raw, dict):
        value: Any = raw.get(
            "value",
            raw.get("theta", raw.get("quality")),
        )
        if value is None:
            raise ValueError(
                "Evaluator dictionaries must include value, theta, or quality."
            )
        token_usage: dict[str, Any] = raw.get("token_usage") or {}
        metadata: dict[str, Any] = {
            key: val
            for key, val in raw.items()
            if key not in {"value", "theta", "quality", "token_usage"}
        }
        numeric_value: float = float(value)
        prompt_tokens: int = int(
            raw.get("prompt_tokens", token_usage.get("prompt_tokens", 0)) or 0
        )
        completion_tokens: int = int(
            raw.get("completion_tokens", token_usage.get("completion_tokens", 0)) or 0
        )
        total_tokens: int = int(
            raw.get("total_tokens", token_usage.get("total_tokens", 0))
            or prompt_tokens + completion_tokens
        )
        return SamplingObservation(
            theta=numeric_value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata=metadata,
        )
    raise TypeError(f"Unsupported evaluator result type: {type(raw)!r}")


def _load_evaluator_class(reference: str) -> type[Evaluator]:
    """Import an evaluator class from a module or module:class reference."""

    module_name, separator, object_name = reference.partition(":")
    module: Any = importlib.import_module(module_name)
    if not separator:
        return _single_evaluator_class_from_module(module, reference)

    try:
        evaluator_object: Any = getattr(module, object_name)
    except AttributeError as exception:
        raise ValueError(
            f"Module '{module_name}' does not define '{object_name}'."
        ) from exception
    return _validate_evaluator_class(evaluator_object, reference)


def _single_evaluator_class_from_module(
    module: Any,
    reference: str,
) -> type[Evaluator]:
    """Return the only concrete Evaluator class defined by a module."""

    evaluator_classes: list[type[Evaluator]] = [
        member
        for _, member in inspect.getmembers(module, inspect.isclass)
        if _is_concrete_evaluator_class(member)
        and member.__module__ == module.__name__
    ]
    if not evaluator_classes:
        raise ValueError(
            f"Module '{reference}' does not define any concrete Evaluator class. "
            "Use a 'module:ClassName' reference."
        )
    if len(evaluator_classes) > 1:
        class_names: list[str] = [cls.__name__ for cls in evaluator_classes]
        raise ValueError(
            f"Module '{reference}' defines more than one concrete Evaluator class: "
            f"{class_names}. Use a 'module:ClassName' reference."
        )
    return evaluator_classes[0]


def _validate_evaluator_class(
    evaluator_object: Any,
    reference: str,
) -> type[Evaluator]:
    """Validate that an imported object is a concrete Evaluator class."""

    if not inspect.isclass(evaluator_object):
        raise TypeError(f"{reference} must point to an Evaluator class.")
    if not _is_evaluator_class(evaluator_object):
        raise TypeError(f"{reference} must inherit from sampling.myevaluators.Evaluator.")
    if inspect.isabstract(evaluator_object):
        raise TypeError(f"{reference} must point to a concrete Evaluator class.")
    return evaluator_object


def _is_concrete_evaluator_class(candidate: Any) -> bool:
    """Return whether a class is a non-abstract Evaluator implementation."""

    return _is_evaluator_class(candidate) and not inspect.isabstract(candidate)


def _is_evaluator_class(candidate: Any) -> bool:
    """Return whether a class inherits from Evaluator, excluding Evaluator itself."""

    return (
        inspect.isclass(candidate)
        and candidate is not Evaluator
        and issubclass(candidate, Evaluator)
    )
