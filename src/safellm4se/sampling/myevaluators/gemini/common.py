"""Shared helpers for Gemini-backed evaluators and HumanEval execution."""

import multiprocessing
import re
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any, ClassVar

from safellm4se.sampling.config import config
from safellm4se.sampling.myevaluators.base_evaluator import BaseEvaluator
from safellm4se.sampling.models import SamplingObservation

DEFAULT_MODEL_ID: str = "gemini-3.1-flash-lite"
DEFAULT_MODEL_NAME: str = "gemini"
API_KEY_ENVIRONMENT_VARIABLE: str = "GEMINI_API_KEY"
DEFAULT_TEST_TIMEOUT: float = 30.0  # Maximum seconds allowed for executing tests.
DEFAULT_MAX_TOKENS: int = 512  # Maximum number of tokens for the LLM response.
HUMANEVAL_PROBLEM_COUNT: int = 164  # Number of problems to be solved.
DEFAULT_SYSTEM_PROMPT: str = ""


class GeminiBaseEvaluator(BaseEvaluator):
    """Base evaluator with shared Gemini configuration and API calls."""

    EXPERIMENT_NAME: ClassVar[str] = ""
    METRIC_TYPE: ClassVar[str] = "continuous"

    def __init__(
        self,
        default_max_tokens: int,
        default_system_prompt: str,
        **parameters: Any,
    ) -> None:
        """Initialize shared Gemini evaluator settings.
        Args:
            default_max_tokens: Default maximum number of generated tokens.
            default_system_prompt: Default system prompt used for generation requests.
            **parameters: Evaluator parameters.
        Returns:
            None.
        Raises:
            Exception: Re-raises any exception produced by parameter conversion.
        """
        super().__init__(**parameters)
        self._set_attribute_from_parameter(
            "_temperature",
            "temperature",
            config.temperature,
            float,
        )  # Sampling temperature used by Gemini.
        self._set_attribute_from_parameter(
            "_model_id",
            "model_id",
            DEFAULT_MODEL_ID,
            str,
        )  # Provider-specific Gemini model identifier.
        self._set_attribute_from_parameter(
            "_model_name",
            "model_name",
            DEFAULT_MODEL_NAME,
            str,
        )  # Canonical model name used in persisted measurements.
        self._set_attribute_from_parameter(
            "_max_tokens",
            "max_tokens",
            default_max_tokens,
            int,
        )  # Maximum number of generated tokens requested from Gemini.
        self._set_attribute_from_parameter(
            "_system_prompt",
            "system_prompt",
            default_system_prompt,
            str,
        )  # System prompt used for generation requests.

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The canonical Gemini-backed model name.
        """
        return self._model_name

    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The evaluator experiment name.
        """
        return self.EXPERIMENT_NAME

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The configured Gemini model identifier.
        """
        return self._model_id

    @property
    def metric_type(self) -> str:
        """Return the statistical variable type used by this evaluator.
        Returns:
            The evaluator metric type.
        Raises:
            None.
        """
        return self.METRIC_TYPE

    def call_gemini(self, prompt: str) -> dict[str, Any]:
        """Call the Gemini API with the configured evaluator settings.
        Args:
            prompt: User prompt sent to Gemini.
        Returns:
            A normalized response dictionary with text and token counters.
        Raises:
            RuntimeError: If the Gemini request fails.
            KeyError: If GEMINI_API_KEY is not configured.
        """
        api_key: str = self._load_api_key_from_environment(
            API_KEY_ENVIRONMENT_VARIABLE
        )
        return call_gemini_generate_content(
            api_key=api_key,
            model_id=self.model_id,
            system_prompt=self._system_prompt,
            user_prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    def build_observation(
        self,
        theta: float,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: dict[str, Any],
    ) -> SamplingObservation:
        """Store run counters and return a sampling observation.
        Args:
            theta: Evaluation result for the run.
            prompt_tokens: Prompt token count reported by Gemini.
            completion_tokens: Completion token count reported by Gemini.
            metadata: Observation metadata.
        Returns:
            A sampling observation populated with shared evaluator fields.
        Raises:
            None.
        """
        self._theta = theta
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        return SamplingObservation(
            theta=self._theta,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            metadata=metadata,
        )


def call_gemini_generate_content(
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Call Gemini generate_content and normalize the SDK response.
    Args:
        api_key: Gemini API key.
        model_id: Provider-specific Gemini model identifier.
        system_prompt: System instruction content.
        user_prompt: User prompt content.
        temperature: Sampling temperature used by Gemini.
        max_tokens: Maximum number of generated tokens requested from Gemini.
    Returns:
        A dictionary containing response text and token usage values.
    Raises:
        RuntimeError: If the Gemini SDK request fails.
        ImportError: If the google-genai SDK is not installed.
    """
    from google import genai
    from google.genai import types

    client: Any = genai.Client(api_key=api_key)
    try:
        response: Any = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
    except Exception as exception:
        raise RuntimeError(f"Gemini request failed: {exception}") from exception

    usage: Any = getattr(response, "usage_metadata", None)
    prompt_tokens: int = _usage_value(usage, "prompt_token_count")
    completion_tokens: int = _usage_value(usage, "candidates_token_count")
    total_tokens: int = (
        _usage_value(usage, "total_token_count")
        or prompt_tokens + completion_tokens
    )
    return {
        "text": _extract_gemini_text(response),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw_response": response,
    }


def _usage_value(usage: Any, field_name: str) -> int:
    """Read an integer usage field from SDK objects or dictionaries.
    Args:
        usage: SDK usage object, usage dictionary, or None.
        field_name: Usage field name to read.
    Returns:
        The integer usage value, or zero when usage is absent.
    Raises:
        ValueError: If the usage value cannot be converted to an integer.
    """
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(field_name) or 0)
    return int(getattr(usage, field_name, 0) or 0)


def _extract_gemini_text(response: Any) -> str:
    """Extract text from Gemini responses across SDK response shapes.
    Args:
        response: Gemini SDK response object.
    Returns:
        The generated text, or an empty string when absent.
    Raises:
        None.
    """
    try:
        text: str | None = getattr(response, "text", None)
    except Exception:
        text = None
    if text:
        return text

    candidates: list[Any] = getattr(response, "candidates", None) or []
    parts_text: list[str] = []
    for candidate in candidates:
        content: Any = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text: str | None = getattr(part, "text", None)
            if part_text:
                parts_text.append(part_text)
    return "\n".join(parts_text)


def response_text(response_data: dict[str, Any]) -> str:
    """Extract assistant content from a normalized Gemini response.
    Args:
        response_data: Normalized Gemini response dictionary.
    Returns:
        The generated text, or an empty string when absent.
    Raises:
        None.
    """
    return str(response_data.get("text") or "")


def response_prompt_tokens(response_data: dict[str, Any]) -> int:
    """Extract prompt token count from a normalized Gemini response.
    Args:
        response_data: Normalized Gemini response dictionary.
    Returns:
        The prompt token count, or zero when absent.
    Raises:
        ValueError: If the token value cannot be converted to an integer.
    """
    return int(response_data.get("prompt_tokens") or 0)


def response_completion_tokens(response_data: dict[str, Any]) -> int:
    """Extract completion token count from a normalized Gemini response.
    Args:
        response_data: Normalized Gemini response dictionary.
    Returns:
        The completion token count, or zero when absent.
    Raises:
        ValueError: If the token value cannot be converted to an integer.
    """
    return int(response_data.get("completion_tokens") or 0)


def sanitize_completion(completion: str) -> str:
    """Extract raw Python code from a model completion.
    Args:
        completion: Text returned by the language model.
    Returns:
        The extracted Python code, without Markdown code fences when present.
    Raises:
        None.
    """
    match: re.Match[str] | None = re.search(
        r"```(?:python)?\s*(.*?)\s*```",
        completion,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return completion.strip()


def load_humaneval_benchmark() -> list[dict[str, Any]]:
    """Load all HumanEval problems in benchmark order.
    Returns:
        A list with all HumanEval dataset items.
    Raises:
        RuntimeError: If the HumanEval dataset dependency cannot be loaded.
    """
    try:
        from datasets import load_dataset
    except ImportError as exception:
        raise RuntimeError(
            "The 'datasets' package is required to load HumanEval."
        ) from exception

    dataset: Any = load_dataset("openai/openai_humaneval", split="test")
    return [dict(dataset_item) for dataset_item in dataset]


def load_humaneval_problem(problem_number: int) -> dict[str, Any]:
    """Load one HumanEval problem by its one-based benchmark number.
    Args:
        problem_number: HumanEval problem number in the inclusive benchmark range.
    Returns:
        The selected HumanEval dataset item.
    Raises:
        ValueError: If the requested problem number is outside the valid range.
        RuntimeError: If the HumanEval dataset dependency cannot be loaded.
    """
    validate_humaneval_problem_number(problem_number)
    return load_humaneval_benchmark()[problem_number - 1]


def validate_humaneval_problem_number(problem_number: int) -> None:
    """Validate a one-based HumanEval problem number.
    Args:
        problem_number: HumanEval problem number to validate.
    Returns:
        None.
    Raises:
        ValueError: If the requested problem number is outside the valid range.
    """
    if not 1 <= problem_number <= HUMANEVAL_PROBLEM_COUNT:
        raise ValueError(
            f"HumanEval problem_number must be between 1 and "
            f"{HUMANEVAL_PROBLEM_COUNT}."
        )


def normalize_humaneval_completion(completion: str, prompt: str) -> str:
    """Normalize completion indentation for HumanEval function prompts.
    Args:
        completion: Python code extracted from the model response.
        prompt: HumanEval prompt that precedes the completion.
    Returns:
        Completion code with prompt-compatible indentation when needed.
    Raises:
        None.
    """
    if not completion:
        return completion

    if _looks_like_complete_python_block(completion):
        return completion

    prompt_lines: list[str] = prompt.splitlines()
    prompt_last_line: str = prompt_lines[-1] if prompt_lines else ""
    prompt_indentation: str = prompt_last_line[
        : len(prompt_last_line) - len(prompt_last_line.lstrip())
    ]
    first_completion_line: str = completion.splitlines()[0]
    if not prompt_indentation or first_completion_line.startswith((" ", "\t")):
        return completion

    return "\n".join(
        f"{prompt_indentation}{completion_line}" if completion_line else ""
        for completion_line in completion.splitlines()
    )


def _looks_like_complete_python_block(completion: str) -> bool:
    """Return whether a completion starts as top-level Python code.
    Args:
        completion: Python code extracted from the model response.
    Returns:
        True when the completion should not be shifted into a function body.
    """
    top_level_prefixes: tuple[str, ...] = (
        "async def ",
        "class ",
        "def ",
        "from ",
        "import ",
        "@",
    )
    for completion_line in completion.splitlines():
        stripped_line: str = completion_line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith("#"):
            continue
        return completion_line == stripped_line and stripped_line.startswith(
            top_level_prefixes
        )
    return False


def _unsafe_execute(
    dataset_item: dict[str, Any],
    completion: str,
    result_queue: Queue,
) -> None:
    """Execute generated code with the HumanEval tests in an isolated scope.
    Args:
        dataset_item: HumanEval item containing prompt, tests, and entry point.
        completion: Python code generated by the language model.
        result_queue: Queue where the test result is stored.
    Returns:
        None.
    Raises:
        None. All execution exceptions are captured as failed results.
    """
    try:
        candidate_code: str = (
            str(dataset_item["prompt"])
            + "\n"
            + "\n"
            + completion
            + "\n"
            + "\n"
            + str(dataset_item["test"])
        )
        exec_globals: dict[str, Any] = {}
        exec(candidate_code, exec_globals)

        entry_point: str = str(dataset_item["entry_point"])
        candidate: Any = exec_globals.get(entry_point)
        check_function: Any = exec_globals.get("check")
        if not callable(candidate):
            raise TypeError(f"Entry point is not callable: {entry_point}")
        if not callable(check_function):
            raise TypeError("HumanEval check function is not callable.")

        candidate_call_count: int = 0

        def counted_candidate(*args: Any, **kwargs: Any) -> Any:
            """Call the generated candidate and count test invocations.
            Args:
                *args: Positional arguments passed by the HumanEval tests.
                **kwargs: Keyword arguments passed by the HumanEval tests.
            Returns:
                The value returned by the generated candidate.
            Raises:
                Exception: Re-raises any exception raised by the candidate.
            """
            nonlocal candidate_call_count
            candidate_call_count += 1
            return candidate(*args, **kwargs)

        check_function(counted_candidate)
        if candidate_call_count == 0:
            raise AssertionError("HumanEval tests did not call the candidate.")
        result_queue.put({"passed": True, "result": "passed"})
    except BaseException as exception:
        result_queue.put(
            {
                "passed": False,
                "result": f"failed: {type(exception).__name__} ({str(exception)})",
            }
        )


def evaluate_sample(
    dataset_item: dict[str, Any],
    completion: str,
    timeout: float,
) -> dict[str, Any]:
    """Evaluate one generated HumanEval solution in a separate process.
    Args:
        dataset_item: HumanEval item containing prompt, tests, and entry point.
        completion: Python code generated by the language model.
        timeout: Maximum number of seconds allowed for test execution.
    Returns:
        A dictionary with task_id, passed, and result keys.
    Raises:
        None.
    """
    result_queue: Queue = multiprocessing.Queue()
    evaluation_process: multiprocessing.Process = multiprocessing.Process(
        target=_unsafe_execute,
        args=(dataset_item, completion, result_queue),
    )
    evaluation_process.start()
    evaluation_process.join(timeout)

    if evaluation_process.is_alive():
        evaluation_process.terminate()
        evaluation_process.join()
        return {
            "task_id": dataset_item["task_id"],
            "passed": False,
            "result": "timed out",
        }

    try:
        result: dict[str, Any] = dict(result_queue.get(timeout=1.0))
    except Empty:
        exit_code: int | None = evaluation_process.exitcode
        return {
            "task_id": dataset_item["task_id"],
            "passed": False,
            "result": f"failed: process exited with code {exit_code}",
        }

    return {
        "task_id": dataset_item["task_id"],
        "passed": bool(result.get("passed", False)),
        "result": str(result.get("result", "failed")),
    }
