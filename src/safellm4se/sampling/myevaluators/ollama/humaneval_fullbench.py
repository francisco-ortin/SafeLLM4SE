"""Ollama evaluator that generates and tests the complete HumanEval benchmark."""

import time
from typing import Any
from loguru import logger

from safellm4se.sampling.myevaluators.ollama.common import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
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
from safellm4se.sampling.models import SamplingObservation

MODEL_ID: str = "deepseek-coder:6.7b" # DEFAULT_MODEL_ID  # Unique universal model identifier.
MODEL_NAME: str = "deepseek-coder" # DEFAULT_MODEL_NAME  # Short model name.
EXPERIMENT_NAME: str = "ollama-humaneval-fullbench"  # Evaluator experiment name.
MAX_TOKENS: int = DEFAULT_MAX_TOKENS  # Maximum number of tokens for each LLM response.
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
        inter_invocation_waiting: float = float(
            context.get("inter_invocation_waiting", 0.0) or 0.0
        )
        benchmark_results: list[dict[str, Any]] = []
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0

        for problem_index, dataset_item in enumerate(
            load_humaneval_benchmark(),
            start=FIRST_PROBLEM_NUMBER,
        ):
            logger.debug(f"Evaluating problem {problem_index} in human eval.")
            result: dict[str, Any] = self._evaluate_problem(
                problem_index,
                dataset_item,
            )
            if inter_invocation_waiting > 0 and problem_index < LAST_PROBLEM_NUMBER:
                logger.debug("Waiting {} seconds before the next HumanEval problem.", inter_invocation_waiting)
                time.sleep(inter_invocation_waiting)
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

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The canonical Ollama-backed model name.
        """
        return MODEL_NAME

    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The evaluator experiment name.
        """
        return EXPERIMENT_NAME

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The configured Ollama model identifier.
        """
        return MODEL_ID

    @property
    def metric_type(self) -> str:
        """Return the statistical variable type used by this evaluator.
        Returns:
            The evaluator metric type.
        """
        return "continuous"

