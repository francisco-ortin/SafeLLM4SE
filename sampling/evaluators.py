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
        """Initialize the evaluator with user-provided parameters.
        Args:
            **parameters: Evaluator-specific constructor parameters.
        """
        self.parameters: dict[str, Any] = parameters

    @abstractmethod
    def run(self, **context: Any) -> SamplingObservation | None:
        """Execute the model call and evaluation for one sampling observation.
        Args:
            **context: Runtime context values supplied by the sampler.
        Returns:
            A sampling observation, or None when the evaluator exposes the result
            through its public properties.
        """

    @property
    @abstractmethod
    def theta(self) -> float:
        """Return the numeric evaluation result for the last run.
        Returns:
            The numeric evaluation result.
        """

    @property
    @abstractmethod
    def metric_type(self) -> str:
        """Return the evaluated variable type.
        Returns:
            The metric type, usually binary or continuous.
        """

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The experiment name.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The canonical model name.
        """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The provider-specific model identifier.
        """

    @property
    @abstractmethod
    def prompt_tokens(self) -> int:
        """Return the number of prompt tokens consumed by the last run.
        Returns:
            The prompt token count.
        """

    @property
    @abstractmethod
    def completion_tokens(self) -> int:
        """Return the number of completion tokens produced by the last run.
        Returns:
            The completion token count.
        """

    @property
    def total_tokens(self) -> int:
        """Return the total token count from the last evaluator run.
        Returns:
            The sum of prompt and completion tokens from the last run.
        """
        return self.completion_tokens + self.prompt_tokens


def load_evaluator(
    reference: str,
    parameters: dict[str, Any] | None = None,
) -> Evaluator:
    """Load and instantiate an evaluator class from a module reference.
    Args:
        reference: Module reference or module:class reference to load.
        parameters: Optional evaluator constructor parameters.
    Returns:
        An instantiated evaluator.
    """
    evaluator_class: type[Evaluator] = _load_evaluator_class(reference)
    return evaluator_class(**(parameters or {}))


def run_evaluator(
    evaluator: Evaluator,
    context: dict[str, Any],
) -> SamplingObservation:
    """Run an evaluator and return its result as a sampling observation.
    Args:
        evaluator: Evaluator instance to run.
        context: Runtime context values passed to the evaluator.
    Returns:
        A normalized sampling observation with model identifiers populated.
    """
    result: SamplingObservation | None = evaluator.run(**context)
    if result is not None:
        observation: SamplingObservation = coerce_observation(result)
    else:
        observation = observation_from_evaluator(evaluator)
    if observation.experiment_name and observation.model_name and observation.model_id:
        return observation
    return replace(
        observation,
        experiment_name=observation.experiment_name or evaluator.experiment_name,
        model_name=observation.model_name or evaluator.model_name,
        model_id=observation.model_id or evaluator.model_id,
    )


def observation_from_evaluator(evaluator: Evaluator) -> SamplingObservation:
    """Build a sampling observation from an evaluator object's public state.
    Args:
        evaluator: Evaluator whose public state contains the latest result.
    Returns:
        A sampling observation built from evaluator properties.
    """
    prompt_tokens: int = evaluator.prompt_tokens
    completion_tokens: int = evaluator.completion_tokens
    theta: float = float(evaluator.theta)
    return SamplingObservation(
        theta=theta,
        experiment_name=evaluator.experiment_name,
        model_name=evaluator.model_name,
        model_id=evaluator.model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        metadata={"evaluator_type": evaluator.metric_type},
    )


def coerce_observation(raw: Any) -> SamplingObservation:
    """Convert supported evaluator return values into a sampling observation.
    Args:
        raw: Raw evaluator return value.
    Returns:
        A sampling observation.
    Raises:
        ValueError: If a dictionary result lacks theta or quality.
        TypeError: If the result type is unsupported.
    """
    if isinstance(raw, SamplingObservation):
        return raw
    if isinstance(raw, bool):
        return SamplingObservation(theta=float(int(raw)))
    if isinstance(raw, (int, float)):
        return SamplingObservation(theta=float(raw))
    if isinstance(raw, dict):
        theta: Any = raw.get("theta", raw.get("quality"))
        if theta is None:
            raise ValueError(
                "Evaluator dictionaries must include theta or quality."
            )
        token_usage: dict[str, Any] = raw.get("token_usage") or {}
        metadata: dict[str, Any] = {
            key: val
            for key, val in raw.items()
            if key not in {"theta", "quality", "experiment_name", "token_usage"}
        }
        numeric_theta: float = float(theta)
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
            theta=numeric_theta,
            experiment_name=str(raw.get("experiment_name", "")),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata=metadata,
        )
    raise TypeError(f"Unsupported evaluator result type: {type(raw)!r}")


def _load_evaluator_class(reference: str) -> type[Evaluator]:
    """Import an evaluator class from a module or module:class reference.
    Args:
        reference: Module reference or module:class reference to load.
    Returns:
        The resolved concrete evaluator class.
    Raises:
        ValueError: If the named class is missing.
    """
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
    """Return the only concrete Evaluator class defined by a module.
    Args:
        module: Imported module to inspect.
        reference: Original module reference used for error messages.
    Returns:
        The only concrete evaluator class defined by the module.
    Raises:
        ValueError: If the module defines zero or multiple concrete evaluators.
    """
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
    """Validate that an imported object is a concrete Evaluator class.
    Args:
        evaluator_object: Imported object to validate.
        reference: Original reference used for error messages.
    Returns:
        The validated concrete evaluator class.
    Raises:
        TypeError: If the object is not a concrete Evaluator class.
    """
    if not inspect.isclass(evaluator_object):
        raise TypeError(f"{reference} must point to an Evaluator class.")
    if not _is_evaluator_class(evaluator_object):
        raise TypeError(
            f"{reference} must inherit from sampling.myevaluators.Evaluator."
        )
    if inspect.isabstract(evaluator_object):
        raise TypeError(f"{reference} must point to a concrete Evaluator class.")
    return evaluator_object


def _is_concrete_evaluator_class(candidate: Any) -> bool:
    """Return whether a class is a non-abstract Evaluator implementation.
    Args:
        candidate: Object to inspect.
    Returns:
        True if the object is a concrete Evaluator subclass; otherwise, False.
    """
    return _is_evaluator_class(candidate) and not inspect.isabstract(candidate)


def _is_evaluator_class(candidate: Any) -> bool:
    """Return whether a class inherits from Evaluator, excluding Evaluator itself.
    Args:
        candidate: Object to inspect.
    Returns:
        True if the object is an Evaluator subclass other than Evaluator itself;
        otherwise, False.
    """
    return (
        inspect.isclass(candidate)
        and candidate is not Evaluator
        and issubclass(candidate, Evaluator)
    )
