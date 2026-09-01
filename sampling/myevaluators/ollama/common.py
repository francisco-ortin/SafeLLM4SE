"""Shared helpers for Ollama-backed evaluators and HumanEval execution."""

import json
import multiprocessing
import re
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from sampling.config import config
from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation

DEFAULT_MODEL_ID: str = "qwen2.5-coder:7b"
DEFAULT_MODEL_NAME: str = "qwen-coder"
DEFAULT_OLLAMA_HOST: str = "http://host.docker.internal:11434"
DEFAULT_TEST_TIMEOUT: float = 30.0  # Maximum seconds allowed for executing tests.
DEFAULT_REQUEST_TIMEOUT: float = 5000.0
DEFAULT_MAX_TOKENS: int = 512  # Maximum number of tokens for the LLM response.
HUMANEVAL_PROBLEM_COUNT: int = 164 # number of problems to be solved
DEFAULT_SYSTEM_PROMPT: str = "You are an expert Python programmer. Complete the given function. "\
                             "Return ONLY the Python code, no explanations, no markdown formatting."

class OllamaBaseEvaluator(BaseEvaluator):
    """Base evaluator with shared Ollama configuration and API calls."""

    EXPERIMENT_NAME: ClassVar[str] = ""
    METRIC_TYPE: ClassVar[str] = "continuous"

    def __init__(
        self,
        default_max_tokens: int,
        default_system_prompt: str,
        **parameters: Any,
    ) -> None:
        """Initialize shared Ollama evaluator settings.
        Args:
            default_max_tokens: Default maximum number of generated tokens.
            default_system_prompt: Default system prompt used for chat requests.
            **parameters: Evaluator parameters.
        Raises:
            Exception: Re-raises any exception produced by parameter conversion.
        """
        super().__init__(**parameters)
        self._set_attribute_from_parameter(
            "_temperature",
            "temperature",
            config.temperature,
            float,
        )  # Sampling temperature used by Ollama.
        self._set_attribute_from_parameter(
            "_model_id",
            "model_id",
            DEFAULT_MODEL_ID,
            str,
        )  # Provider-specific Ollama model identifier.
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
        )  # Maximum number of generated tokens requested from Ollama.
        self._set_attribute_from_parameter(
            "_ollama_host",
            "ollama_host",
            DEFAULT_OLLAMA_HOST,
            str,
        )  # Base URL of the Ollama API.
        self._set_attribute_from_parameter(
            "_system_prompt",
            "system_prompt",
            default_system_prompt,
            str,
        )  # System prompt used for chat requests.

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The canonical Ollama-backed model name.
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
            The configured Ollama model identifier.
        """
        return self._model_id

    @property
    def metric_type(self) -> str:
        """Return the statistical variable type used by this evaluator.
        Returns:
            The evaluator metric type.
        """
        return self.METRIC_TYPE

    def call_ollama(self, prompt: str) -> dict[str, Any]:
        """Call the Ollama chat API with the configured evaluator settings.
        Args:
            prompt: User prompt sent to Ollama.
        Returns:
            The decoded JSON response returned by Ollama.
        Raises:
            RuntimeError: If Ollama returns an HTTP error or cannot be reached.
        """
        return call_ollama_chat(
            host=self._ollama_host,
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
            prompt_tokens: Prompt token count reported by Ollama.
            completion_tokens: Completion token count reported by Ollama.
            metadata: Observation metadata.
        Returns:
            A sampling observation populated with shared evaluator fields.
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


def call_ollama_chat(
    host: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Call the Ollama chat API and decode the JSON response.
    Args:
        host: Base URL of the Ollama API.
        model_id: Provider-specific Ollama model identifier.
        system_prompt: System message content.
        user_prompt: User message content.
        temperature: Sampling temperature used by Ollama.
        max_tokens: Maximum number of generated tokens requested from Ollama.
    Returns:
        The decoded JSON response returned by Ollama.
    Raises:
        RuntimeError: If Ollama returns an HTTP error or cannot be reached.
    """
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": int(max_tokens),
        },
        "stream": False,
    }
    request: Request = Request(
        urljoin(host.rstrip("/") + "/", "api/chat"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=DEFAULT_REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exception:
        raise RuntimeError(
            f"Ollama request failed with HTTP {exception.code}"
        ) from exception
    except URLError as exception:
        raise RuntimeError(f"Ollama is not reachable at {host}") from exception


def response_text(response_data: dict[str, Any]) -> str:
    """Extract assistant content from an Ollama chat response.
    Args:
        response_data: Decoded Ollama JSON response.
    Returns:
        The assistant message content, or an empty string when absent.
    Raises:
        None.
    """
    return str((response_data.get("message") or {}).get("content") or "")


def response_prompt_tokens(response_data: dict[str, Any]) -> int:
    """Extract prompt token count from an Ollama response.
    Args:
        response_data: Decoded Ollama JSON response.
    Returns:
        The prompt token count, or zero when absent.
    Raises:
        None.
    """
    return int(response_data.get("prompt_eval_count") or 0)


def response_completion_tokens(response_data: dict[str, Any]) -> int:
    """Extract completion token count from an Ollama response.
    Args:
        response_data: Decoded Ollama JSON response.
    Returns:
        The completion token count, or zero when absent.
    Raises:
        None.
    """
    return int(response_data.get("eval_count") or 0)


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
        candidate_call_count: int = 0
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
    # TODO: Aquí es donde tenemos que comprobar si las pruebas se están ejecutando bien o no
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
