"""Run SAFE-style LLM code-generation evaluations on HumanEval tasks."""

import multiprocessing
import os
import re
import time
from collections.abc import MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
from datasets import Dataset, load_dataset
from loguru import logger
from statsmodels.stats.proportion import proportion_confint

from config.config import Config
from config.logger import setup_logger
from csvutils import (
    append_measurements_csv,
    build_raw_results_from_measurements,
    create_measurement_row,
    get_completed_execution_numbers,
    get_results_file_paths,
    read_measurements_csv,
)
from llms import (
    LLMClient,
    LLMCompletion,
    LLMQuotaExceededError,
    LLMRateLimitError,
    create_llm_client,
)
from results import compute_results

MODEL_TO_USE = "glm"  # set the model ID to run this experiment

CURRENT_LOG_FILE: Optional[str] = setup_logger(
    console_level="INFO",
    file_level="TRACE",
    auto_log_file=True,
)
PROJECT_ROOT = Path(__file__).resolve().parent
RATE_LIMIT_LOCK_PATH = PROJECT_ROOT / ".llm_rate_limit.lock"
RATE_LIMIT_TIMESTAMP_PATH = PROJECT_ROOT / ".llm_rate_limit.timestamp"


@contextmanager
def interprocess_file_lock(lock_path: Path) -> Generator[None, None, None]:
    """Serializes LLM invocations across independent Python processes."""
    lock_file = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def wait_for_global_llm_slot() -> None:
    """Applies Config.INTER_INVOCATIONS_TIMEOUT across all running processes."""
    with interprocess_file_lock(RATE_LIMIT_LOCK_PATH):
        last_invocation: float = 0.0
        if RATE_LIMIT_TIMESTAMP_PATH.exists():
            try:
                last_invocation = float(
                    RATE_LIMIT_TIMESTAMP_PATH.read_text(encoding="utf-8")
                )
            except ValueError:
                logger.warning("Ignoring invalid LLM rate-limit timestamp.")

        elapsed: float = time.time() - last_invocation
        wait_seconds: float = Config.INTER_INVOCATIONS_WAITING - elapsed
        if wait_seconds > 0:
            logger.info(f"Waiting {wait_seconds:.1f}s before next LLM invocation.")
            time.sleep(wait_seconds)

        RATE_LIMIT_TIMESTAMP_PATH.write_text(str(time.time()), encoding="utf-8")


def sanitize_completion(completion: str) -> str:
    """Extracts raw Python code if the LLM response is wrapped in Markdown blocks."""
    match: re.Match[str] | None = re.search(
        r"```(?:python)?\s*(.*?)\s*```",
        completion,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return completion.strip()


def request_llm_completion_with_quota_retry(
    llm_client: LLMClient,
    prompt: str,
) -> LLMCompletion:
    """Request one LLM completion, waiting indefinitely when quota is exhausted.

    Args:
        llm_client: Provider-specific LLM client used for the request.
        prompt: HumanEval prompt to send as the user message.

    Returns:
        The completion returned by the LLM provider.
    """
    while True:
        try:
            wait_for_global_llm_slot()
            return llm_client.generate_completion(
                system_prompt=Config.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=Config.TEMPERATURE,
                max_tokens=256,
            )
        except (LLMQuotaExceededError, LLMRateLimitError) as exception:
            waiting_seconds: float = Config.LLM_QUOTA_REACHED_WAITING
            logger.warning(
                "LLM quota or rate limit reached. "
                f"Waiting {waiting_seconds:.1f}s before retrying. "
                f"Exception: {exception}"
            )
            time.sleep(waiting_seconds)


def _unsafe_execute(
    dataset_item: dict[str, Any],
    completion: str,
    result_dict: MutableMapping[str, Any],
) -> None:
    """Execute generated code with its prompt and unit tests in an isolated scope."""
    try:
        full_code: str = (
            dataset_item["prompt"] + "\n" +
            completion + "\n" +
            dataset_item["test"] + "\n" +
            f"check({dataset_item['entry_point']})"
        )
        exec_globals: dict[str, Any] = {}
        exec(full_code, exec_globals)
        result_dict["passed"] = True
        result_dict["result"] = "passed"
    except Exception as exception:
        result_dict["passed"] = False
        result_dict["result"] = (
            f"failed: {type(exception).__name__} ({str(exception)})"
        )


def evaluate_sample_windows(
    dataset_item: dict[str, Any],
    completion: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Evaluate a code sample in a separate process with a timeout."""
    manager = multiprocessing.Manager()
    result_dict: MutableMapping[str, Any] = manager.dict()
    result_dict["passed"] = False
    result_dict["result"] = "timed out"

    evaluation_process = multiprocessing.Process(
        target=_unsafe_execute,
        args=(dataset_item, completion, result_dict),
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

    return {
        "task_id": dataset_item["task_id"],
        "passed": result_dict.get("passed", False),
        "result": result_dict.get("result", "failed"),
    }


def run_model(
    dataset: Dataset,
    llm_client: LLMClient,
    model_name: str,
    existing_measurements: list[dict[str, Any]],
    measurements_path: str,
) -> list[dict[str, Any]]:
    """Runs the LLM against benchmark problems for a specified number of samples."""
    all_measurements: list[dict[str, Any]] = list(existing_measurements)
    for problem in list(dataset)[:Config.N_PROBLEMS]:
        task_id: str = problem["task_id"]
        prompt: str = problem["prompt"]
        completed_executions: set[int] = get_completed_execution_numbers(
            all_measurements,
            task_id,
            model_name,
        )

        logger.info(f"Evaluating {task_id} for '{model_name}'...")
        if completed_executions:
            logger.info(
                f"Skipping {len(completed_executions)} "
                f"existing samples for {task_id}."
            )
        if len(completed_executions) >= Config.N_SAMPLES:
            continue

        logger.debug(f"Prompt for {task_id}:\n{prompt}\n")
        for execution_number in range(1, Config.N_SAMPLES + 1):
            if execution_number in completed_executions:
                continue

            llm_completion: LLMCompletion = request_llm_completion_with_quota_retry(
                llm_client,
                prompt,
            )

            completion: str = sanitize_completion(llm_completion.text)

            logger.debug(
                f"Completion for {task_id}[{execution_number}]:\n{completion}"
            )

            evaluation_result: dict[str, Any] = evaluate_sample_windows(
                problem,
                completion,
                timeout=Config.TEST_TIMEOUT,
            )
            logger.debug(
                f"Result for {task_id}[{execution_number}]: {evaluation_result}"
            )
            status: str = "Passed" if evaluation_result["passed"] else "Failed"
            logger.info(
                f"Executed sample {execution_number}/{Config.N_SAMPLES} "
                f"of {task_id}, model '{model_name}', temperature = {Config.TEMPERATURE}: {status}"
            )
            passed: int = 1 if evaluation_result["passed"] else 0

            measurement: dict[str, Any] = create_measurement_row(
                task_id,
                execution_number,
                passed,
                llm_completion.token_usage,
                model_name,
            )
            append_measurements_csv([measurement], measurements_path)
            all_measurements.append(measurement)
            completed_executions.add(execution_number)

    return all_measurements


def evaluate_results(raw_results: dict[tuple[str, str, str], list[int]]) -> None:
    """Compute and display statistical indicators using the SAFE framework."""
    logger.info("=== SAFE FRAMEWORK REPORT ===")
    for (task_id, model, temperature), passes in raw_results.items():
        successes: int = sum(passes)
        n: int = len(passes)
        if n == 0:
            continue

        quality_p: float = successes / n

        stability_sd: float = float(np.std(passes, ddof=1)) if n > 1 else 0.0

        ci_low, ci_high = proportion_confint(
            successes,
            n,
            method="wilson",
            alpha=0.05,
        )

        logger.info(f"Task: {task_id}")
        logger.info(f"  - Model:                            {model}")
        logger.info(f"  - Temperature:                      {temperature}")
        logger.info(f"  - Quality (Success rate):             {quality_p:.2f}")
        logger.info(f"  - Stability (Std Dev):               {stability_sd:.2f}")
        logger.info(
            "  - Estimation Uncertainty (95% CI):   "
            f"[{ci_low:.2f}, {ci_high:.2f}]"
        )


def main() -> None:
    """Main execution function."""
    model_name: str = Config.get_model_name(MODEL_TO_USE)
    client: LLMClient = create_llm_client(MODEL_TO_USE, model_name, "api-keys.json")

    dataset: Dataset = load_dataset("openai/openai_humaneval", split="test")
    measurements_path, results_path = get_results_file_paths(CURRENT_LOG_FILE)
    existing_measurements: list[dict[str, Any]] = read_measurements_csv(
        measurements_path
    )

    all_measurements: list[dict[str, Any]] = run_model(
        dataset,
        client,
        model_name,
        existing_measurements,
        measurements_path,
    )
    compute_results(measurements_path, results_path)
    raw_results: dict[tuple[str, str, str], list[int]] = (
        build_raw_results_from_measurements(all_measurements)
    )

    evaluate_results(raw_results)

    logger.info(f"Measurements written to {measurements_path}")
    logger.info(f"Summary results written to {results_path}")


if __name__ == "__main__":
    main()
