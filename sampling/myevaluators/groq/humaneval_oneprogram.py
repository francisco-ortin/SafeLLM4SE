"""Groq evaluator that generates and tests one fixed HumanEval program."""

from typing import Any

from sampling.myevaluators.groq.common import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEST_TIMEOUT,
    GroqBaseEvaluator,
    evaluate_sample,
    load_humaneval_problem,
    normalize_humaneval_completion,
    response_completion_tokens,
    response_prompt_tokens,
    response_text,
    sanitize_completion,
    validate_humaneval_problem_number,
)
from sampling.models import SamplingObservation

MODEL_ID: str = DEFAULT_MODEL_ID  # Unique universal model identifier.
MODEL_NAME: str = DEFAULT_MODEL_NAME  # Short model name.
EXPERIMENT_NAME: str = "groq-humaneval-oneprogram"  # Evaluator experiment name.
MAX_TOKENS: int = DEFAULT_MAX_TOKENS  # Maximum number of tokens for each LLM response.
PROBLEM_NUMBER: int = 1  # HumanEval problem number.
TEST_TIMEOUT: float = DEFAULT_TEST_TIMEOUT  # Maximum seconds allowed for tests.
SYSTEM_PROMPT: str = (
    "You are a helpful assistant that only writes valid Python code. "
    "Do not include any other text, explanation, or formatting in your response."
)


class GroqHumanEvalOneProgramEvaluator(GroqBaseEvaluator):
    """Evaluator that tests one Groq-generated Python program with HumanEval."""

    EXPERIMENT_NAME: str = EXPERIMENT_NAME  # Experiment represented by the class.
    METRIC_TYPE: str = "binary"  # Statistical metric type.

    def __init__(self, **parameters: Any) -> None:
        """Initialize the evaluator with model, task, and timeout settings.
        Args:
            **parameters: Evaluator parameters. Defaults are provided if not specified.
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
            RuntimeError: If Groq request fails or HumanEval cannot be loaded.
            FileNotFoundError: If the API key file does not exist.
            KeyError: If no Groq API key is configured.
        """
        del context
        dataset_item: dict[str, Any] = load_humaneval_problem(self._problem_number)
        prompt: str = str(dataset_item["prompt"])
        response_data: dict[str, Any] = self.call_groq(prompt)
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
        theta: float = float(1 if evaluation_result["passed"] else 0)

        return self.build_observation(
            theta=theta,
            prompt_tokens=response_prompt_tokens(response_data),
            completion_tokens=response_completion_tokens(response_data),
            metadata={
                "provider": "groq",
                "task_id": dataset_item["task_id"],
                "problem_number": self._problem_number,
                "entry_point": dataset_item["entry_point"],
                "passed": evaluation_result["passed"],
                "result": evaluation_result["result"],
                "raw_text": raw_text,
                "completion": completion,
            },
        )
