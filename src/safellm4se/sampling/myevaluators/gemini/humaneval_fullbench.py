"""Gemini evaluator that generates and tests the complete HumanEval benchmark."""

import time
from typing import Any

from loguru import logger

from safellm4se.sampling.config import config
from safellm4se.sampling.myevaluators.gemini.common import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEST_TIMEOUT,
    HUMANEVAL_PROBLEM_COUNT,
    GeminiBaseEvaluator,
    evaluate_sample,
    load_humaneval_benchmark,
    normalize_humaneval_completion,
    response_completion_tokens,
    response_prompt_tokens,
    response_text,
    sanitize_completion,
)
from safellm4se.sampling.models import SamplingObservation

MODEL_ID: str = DEFAULT_MODEL_ID  # Unique universal model identifier.
MODEL_NAME: str = DEFAULT_MODEL_NAME  # Short model name.
EXPERIMENT_NAME: str = "gemini-humaneval-fullbench"  # Evaluator experiment name.
MAX_TOKENS: int = DEFAULT_MAX_TOKENS  # Maximum number of tokens for each LLM response.
FIRST_PROBLEM_NUMBER: int = 1  # First one-based HumanEval problem number.
LAST_PROBLEM_NUMBER: int = HUMANEVAL_PROBLEM_COUNT  # Last HumanEval problem number.
TEST_TIMEOUT: float = DEFAULT_TEST_TIMEOUT  # Maximum seconds allowed for each test.
SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT  # Generate Python code only.


def _is_gemini_quota_or_rate_limit_error(exception: RuntimeError) -> bool:
    """Return whether a RuntimeError represents a recoverable Gemini request error.
    Args:
        exception: Runtime error raised by the Gemini request wrapper.
    Returns:
        True when the error message indicates a recoverable Gemini throttling or
        temporary availability condition; otherwise, False.
    """
    message: str = str(exception).casefold()
    return (
        "429" in message
        or "503" in message
        or "resource_exhausted" in message
        or "unavailable" in message
        or "quota exceeded" in message
        or "high demand" in message
        or "rate limit" in message
    )


class GeminiHumanEvalFullBenchEvaluator(GeminiBaseEvaluator):
    """Evaluator that tests Gemini-generated Python programs with HumanEval."""

    EXPERIMENT_NAME: str = EXPERIMENT_NAME  # Experiment represented by the class.
    METRIC_TYPE: str = "continuous"  # Statistical metric type.

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model and timeout settings.
        Args:
            **parameters: Evaluator parameters. Defaults are provided if not specified.
        Returns:
            None.
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
            **context: Runtime context values, including the optional
                inter-invocation delay.
        Returns:
            A sampling observation whose theta is the proportion of HumanEval
            programs that passed their tests.
        Raises:
            RuntimeError: If a non-recoverable Gemini request fails, HumanEval
                cannot be loaded, or all problems are skipped by recoverable
                Gemini request errors.
            FileNotFoundError: If the API key file does not exist.
            KeyError: If no Gemini API key is configured.
            ValueError: If a waiting time cannot be converted to float.
        """
        inter_invocation_waiting: float = float(
            context.get("inter_invocation_waiting", 0.0) or 0.0
        )
        reservation_ttl_seconds: float = float(
            context.get(
                "reservation_ttl_seconds",
                config.reservation_ttl_seconds,
            )
            or 0.0
        )
        benchmark_results: list[dict[str, Any]] = []
        skipped_results: list[dict[str, Any]] = []
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0

        for problem_index, dataset_item in enumerate(
            load_humaneval_benchmark(),
            start=FIRST_PROBLEM_NUMBER,
        ):
            logger.debug(f"Evaluating problem {problem_index} in human eval.")
            try:
                result: dict[str, Any] = self._evaluate_problem(
                    problem_index,
                    dataset_item,
                )
            except RuntimeError as exception:
                logger.debug(f"Exception in problem {problem_index}: {exception}")
                if not _is_gemini_quota_or_rate_limit_error(exception):
                    raise
                skipped_result: dict[str, Any] = {
                    "task_id": dataset_item["task_id"],
                    "problem_number": problem_index,
                    "entry_point": dataset_item["entry_point"],
                    "reason": str(exception),
                }
                skipped_results.append(skipped_result)
                logger.warning(
                    "Skipping HumanEval problem {} after recoverable Gemini "
                    "request error. Waiting {} seconds before the next problem.",
                    problem_index,
                    reservation_ttl_seconds,
                )
                if reservation_ttl_seconds > 0 and problem_index < LAST_PROBLEM_NUMBER:
                    logger.debug(
                        f"Waiting {reservation_ttl_seconds} seconds before "
                        "next execution"
                    )
                    time.sleep(reservation_ttl_seconds)
                continue
            if inter_invocation_waiting > 0 and problem_index < LAST_PROBLEM_NUMBER:
                logger.debug(
                    "Waiting {} seconds before the next HumanEval problem.",
                    inter_invocation_waiting,
                )
                time.sleep(inter_invocation_waiting)
            total_prompt_tokens += int(result.pop("prompt_tokens"))
            total_completion_tokens += int(result.pop("completion_tokens"))
            benchmark_results.append(result)

        if not benchmark_results:
            raise RuntimeError(
                "All HumanEval problems were skipped because recoverable Gemini "
                "request errors prevented evaluation."
            )

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
                "provider": "gemini",
                "first_problem_number": FIRST_PROBLEM_NUMBER,
                "last_problem_number": LAST_PROBLEM_NUMBER,
                "passed_programs": passed_programs,
                "total_programs": total_programs,
                "skipped_programs": len(skipped_results),
                "skipped_results": skipped_results,
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
            RuntimeError: If the Gemini request fails.
            FileNotFoundError: If the API key file does not exist.
            KeyError: If no Gemini API key is configured.
        """
        prompt: str = str(dataset_item["prompt"])
        response_data: dict[str, Any] = self.call_gemini(prompt)
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
