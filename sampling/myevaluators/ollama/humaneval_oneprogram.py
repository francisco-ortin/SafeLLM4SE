"""Ollama evaluator that generates and tests one fixed HumanEval program."""

import json
import multiprocessing
import re
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from sampling.config import config
from sampling.myevaluators.base_evaluator import BaseEvaluator
from sampling.models import SamplingObservation

# Values of parameters if not passed in the command line
MODEL_ID: str = "qwen2.5-coder:7b"  # Unique universal model identifier.
MODEL_NAME: str = "qwen-coder"  # Short model name.
# Name of the experiment represented by this evaluator.
EXPERIMENT_NAME: str = "ollama-humaneval-oneprogram"
MAX_TOKENS: int = 512  # Maximum number of tokens for the LLM response.
OLLAMA_HOST: str = "http://host.docker.internal:11434"  # Host API for Ollama.
PROBLEM_NUMBER: int = 1  # HumanEval problem number from 1 to 164.
TEST_TIMEOUT: float = 30.0  # Maximum seconds allowed for executing tests.
SYSTEM_PROMPT: str = (
    "Complete the HumanEval Python programming task. "
    "Return only Python code, without Markdown fences or explanations."
)


class OllamaHumanEvalOneProgramEvaluator(BaseEvaluator):
    """Evaluator that tests one Ollama-generated Python program with HumanEval."""

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model, task, and timeout settings.
        Args:
            **parameters: Evaluator parameters. Defaults are provided if not specified.
        Raises:
            ValueError: If the configured HumanEval problem number is invalid.
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
            MODEL_ID,
            str,
        )  # Provider-specific Ollama model identifier.
        self._set_attribute_from_parameter(
            "_model_name",
            "model_name",
            MODEL_NAME,
            str,
        )  # Canonical model name used in persisted measurements.
        self._set_attribute_from_parameter(
            "_max_tokens",
            "max_tokens",
            MAX_TOKENS,
            int,
        )  # Maximum number of generated tokens requested from Ollama.
        self._set_attribute_from_parameter(
            "_ollama_host",
            "ollama_host",
            OLLAMA_HOST,
            str,
        )  # Base URL of the Ollama API.
        self._set_attribute_from_parameter(
            "_problem_number",
            "problem_number",
            PROBLEM_NUMBER,
            int,
        )  # One-based HumanEval problem number selected for evaluation.
        self._set_attribute_from_parameter(
            "_test_timeout",
            "test_timeout",
            TEST_TIMEOUT,
            float,
        )  # Maximum seconds allowed for HumanEval test execution.
        self._set_attribute_from_parameter(
            "_system_prompt",
            "system_prompt",
            SYSTEM_PROMPT,
            str,
        )  # System prompt used for code generation.
        if not 1 <= self._problem_number <= 164:
            raise ValueError("HumanEval problem_number must be between 1 and 164.")

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The canonical Ollama-backed model name.
        Raises:
            None.
        """
        return self._model_name

    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The Ollama HumanEval experiment name.
        Raises:
            None.
        """
        return EXPERIMENT_NAME

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The configured Ollama model identifier.
        Raises:
            None.
        """
        return self._model_id

    @property
    def metric_type(self) -> str:
        """Return the binary variable type used by this evaluator.
        Returns:
            The binary metric type.
        Raises:
            None.
        """
        return "binary"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Generate one HumanEval solution, test it, and update evaluator state.
        Args:
            **context: Runtime context values. The evaluator uses constructor
                parameters for the HumanEval problem and Ollama settings.
        Returns:
            A sampling observation whose theta is 1.0 for pass and 0.0 for fail.
        Raises:
            RuntimeError: If Ollama returns an HTTP error, cannot be reached, or
                the HumanEval dataset cannot be loaded.
        """
        del context
        dataset_item: dict[str, Any] = load_humaneval_problem(self._problem_number)
        response_data: dict[str, Any] = self._call_ollama(dataset_item["prompt"])
        raw_text: str = str((response_data.get("message") or {}).get("content") or "")
        completion: str = sanitize_completion(raw_text)
        evaluation_result: dict[str, Any] = evaluate_sample(
            dataset_item,
            completion,
            self._test_timeout,
        )

        self._prompt_tokens = int(response_data.get("prompt_eval_count") or 0)
        self._completion_tokens = int(response_data.get("eval_count") or 0)
        self._theta = float(1 if evaluation_result["passed"] else 0)

        return SamplingObservation(
            theta=self._theta,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            metadata={
                "provider": "ollama",
                "task_id": dataset_item["task_id"],
                "problem_number": self._problem_number,
                "entry_point": dataset_item["entry_point"],
                "passed": evaluation_result["passed"],
                "result": evaluation_result["result"],
                "raw_text": raw_text,
                "completion": completion,
            },
        )

    def _call_ollama(self, prompt: str) -> dict[str, Any]:
        """Call the Ollama chat API for one HumanEval prompt.
        Args:
            prompt: HumanEval prompt to send as the user message.
        Returns:
            The decoded JSON response returned by Ollama.
        Raises:
            RuntimeError: If Ollama returns an HTTP error or cannot be reached.
        """
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": self._temperature,
                "num_predict": int(self._max_tokens),
            },
            "stream": False,
        }
        request = Request(
            urljoin(self._ollama_host.rstrip("/") + "/", "api/chat"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5000.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exception:
            raise RuntimeError(
                f"Ollama request failed with HTTP {exception.code}"
            ) from exception
        except URLError as exception:
            raise RuntimeError(
                f"Ollama is not reachable at {self._ollama_host}"
            ) from exception


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


def load_humaneval_problem(problem_number: int) -> dict[str, Any]:
    """Load one HumanEval problem by its one-based benchmark number.
    Args:
        problem_number: HumanEval problem number in the inclusive range 1..164.
    Returns:
        The selected HumanEval dataset item.
    Raises:
        ValueError: If the requested problem number is outside the valid range.
        RuntimeError: If the HumanEval dataset dependency cannot be loaded.
    """
    if not 1 <= problem_number <= 164:
        raise ValueError("HumanEval problem_number must be between 1 and 164.")

    try:
        from datasets import load_dataset
    except ImportError as exception:
        raise RuntimeError(
            "The 'datasets' package is required to load HumanEval."
        ) from exception

    dataset: Any = load_dataset("openai/openai_humaneval", split="test")
    return dict(dataset[problem_number - 1])


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
            dataset_item["prompt"]
            + "\n"
            + completion
            + "\n"
            + dataset_item["test"]
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
            """Call the generated candidate and count HumanEval test invocations.
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
