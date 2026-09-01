"""Ollama evaluator that generates and tests one fixed HumanEval program."""

from typing import Any

from sampling.myevaluators.ollama.common import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TEST_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    HUMANEVAL_PROBLEM_COUNT,
    OllamaBaseEvaluator,
    evaluate_sample,
    load_humaneval_problem,
    response_completion_tokens,
    response_prompt_tokens,
    response_text,
    sanitize_completion,
    validate_humaneval_problem_number,
)
from sampling.models import SamplingObservation

MODEL_ID: str = DEFAULT_MODEL_ID  # Unique universal model identifier.
MODEL_NAME: str = DEFAULT_MODEL_NAME  # Short model name.
EXPERIMENT_NAME: str = "ollama-humaneval-oneprogram"  # Evaluator experiment name.
MAX_TOKENS: int = DEFAULT_MAX_TOKENS  # Maximum number of tokens for each LLM response.
OLLAMA_HOST: str = DEFAULT_OLLAMA_HOST  # Host API for Ollama.
PROBLEM_NUMBER: int = 1  # HumanEval problem number.
TEST_TIMEOUT: float = DEFAULT_TEST_TIMEOUT  # Maximum seconds allowed for executing tests.
SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT  # Generate Python code only.


class OllamaHumanEvalOneProgramEvaluator(OllamaBaseEvaluator):
    """Evaluator that tests one Ollama-generated Python program with HumanEval."""

    EXPERIMENT_NAME: str = EXPERIMENT_NAME
    METRIC_TYPE: str = "binary"

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model, task, and timeout settings.
        Args:
            **parameters: Evaluator parameters. Defaults are provided if not specified.
        Returns:
            None.
        Raises:
            ValueError: If the configured HumanEval problem number is invalid.
            Exception: Re-raises any exception produced by parameter conversion.
        """
        super().__init__(
            default_max_tokens=MAX_TOKENS,
            default_system_prompt=SYSTEM_PROMPT,
            **parameters,
        )
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
        validate_humaneval_problem_number(self._problem_number)

    def run(self, **context: Any) -> SamplingObservation | None:
        """Generate one HumanEval solution, test it, and update evaluator state.
        Args:
            **context: Runtime context values. They are ignored by this evaluator.
        Returns:
            A sampling observation whose theta is 1.0 for pass and 0.0 for fail.
        Raises:
            RuntimeError: If Ollama returns an HTTP error, cannot be reached, or
                the HumanEval dataset cannot be loaded.
        """
        del context
        dataset_item: dict[str, Any] = load_humaneval_problem(self._problem_number)
        response_data: dict[str, Any] = self.call_ollama(str(dataset_item["prompt"]))
        raw_text: str = response_text(response_data)
        completion: str = sanitize_completion(raw_text)
        evaluation_result: dict[str, Any] = evaluate_sample(
            dataset_item,
            completion,
            self._test_timeout,
        )
        theta: float = float(1 if evaluation_result["passed"] else 0)

        return self.build_observation(
            theta=theta,
            prompt_tokens=response_prompt_tokens(response_data),
            completion_tokens=response_completion_tokens(response_data),
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
