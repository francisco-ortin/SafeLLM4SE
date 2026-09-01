"""Ollama evaluator that generates and tests the complete HumanEval benchmark."""

from typing import Any

from sampling.myevaluators.ollama.common import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TEST_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
    HUMANEVAL_PROBLEM_COUNT,
    OllamaBaseEvaluator,
    evaluate_sample,
    load_humaneval_benchmark,
    normalize_humaneval_completion,
    response_completion_tokens,
    response_prompt_tokens,
    response_text,
    sanitize_completion,
    DEFAULT_MAX_TOKENS,
)
from sampling.models import SamplingObservation

MODEL_ID: str = DEFAULT_MODEL_ID  # Unique universal model identifier.
MODEL_NAME: str = DEFAULT_MODEL_NAME  # Short model name.
EXPERIMENT_NAME: str = "ollama-humaneval-fullbench"  # Evaluator experiment name.
MAX_TOKENS: int = DEFAULT_MAX_TOKENS  # Maximum number of tokens for each LLM response.
OLLAMA_HOST: str = DEFAULT_OLLAMA_HOST  # Host API for Ollama.
FIRST_PROBLEM_NUMBER: int = 1  # First one-based HumanEval problem number.
LAST_PROBLEM_NUMBER: int = HUMANEVAL_PROBLEM_COUNT  # Last HumanEval problem number.
TEST_TIMEOUT: float = DEFAULT_TEST_TIMEOUT  # Maximum seconds allowed for each test.
SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT  # Generate Python code only.


class OllamaHumanEvalFullBenchEvaluator(OllamaBaseEvaluator):
    """Evaluator that tests Ollama-generated Python programs with HumanEval."""

    EXPERIMENT_NAME: str = EXPERIMENT_NAME
    METRIC_TYPE: str = "continuous"

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model and timeout settings.
        Args:
            **parameters: Evaluator parameters. Defaults are provided if not specified.
        Raises:
            Exception: Re-raises any exception produced by parameter conversion.
        """
        super().__init__(
            default_max_tokens=MAX_TOKENS,
            default_system_prompt=SYSTEM_PROMPT,
            **parameters,
        )
        self._set_attribute_from_parameter(
            "_test_timeout",
            "test_timeout",
            TEST_TIMEOUT,
            float,
        )  # Maximum seconds allowed for each HumanEval test execution.

    def run(self, **context: Any) -> SamplingObservation | None:
        """Generate all HumanEval solutions, test them, and aggregate pass rate.
        Args:
            **context: Runtime context values. They are ignored by this evaluator.
        Returns:
            A sampling observation whose theta is the proportion of HumanEval
            programs that passed their tests.
        Raises:
            RuntimeError: If Ollama returns an HTTP error, cannot be reached, or
                the HumanEval dataset cannot be loaded.
        """
        del context
        benchmark_results: list[dict[str, Any]] = []
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0

        for problem_index, dataset_item in enumerate(
            load_humaneval_benchmark(),
            start=FIRST_PROBLEM_NUMBER,
        ):
            result: dict[str, Any] = self._evaluate_problem(
                problem_index,
                dataset_item,
            )
            total_prompt_tokens += int(result.pop("prompt_tokens"))
            total_completion_tokens += int(result.pop("completion_tokens"))
            benchmark_results.append(result)

        passed_programs: int = sum(
            1 for benchmark_result in benchmark_results if benchmark_result["passed"]
        )
        total_programs: int = len(benchmark_results)
        theta: float = float(passed_programs / total_programs)

        return self.build_observation(
            theta=theta,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            metadata={
                "provider": "ollama",
                "first_problem_number": FIRST_PROBLEM_NUMBER,
                "last_problem_number": LAST_PROBLEM_NUMBER,
                "passed_programs": passed_programs,
                "total_programs": total_programs,
                "results": benchmark_results,
            },
        )

    def _evaluate_problem(
        self,
        problem_index: int,
        dataset_item: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate and test a solution for one HumanEval problem.
        Args:
            problem_index: One-based HumanEval problem number.
            dataset_item: HumanEval dataset item.
        Returns:
            A benchmark result dictionary with generated code and token counts.
        Raises:
            RuntimeError: If Ollama returns an HTTP error or cannot be reached.
        """
        prompt: str = str(dataset_item["prompt"])
        response_data: dict[str, Any] = self.call_ollama(prompt)
        raw_text: str = response_text(response_data)
        completion: str = normalize_humaneval_completion(
            sanitize_completion(raw_text),
            prompt,
        )
        evaluation_result: dict[str, Any] = evaluate_sample(
            dataset_item,
            completion,
            self._test_timeout,
        )

        return {
            "task_id": dataset_item["task_id"],
            "problem_number": problem_index,
            "entry_point": dataset_item["entry_point"],
            "passed": evaluation_result["passed"],
            "result": evaluation_result["result"],
            "raw_text": raw_text,
            "completion": completion,
            "prompt_tokens": response_prompt_tokens(response_data),
            "completion_tokens": response_completion_tokens(response_data),
        }
