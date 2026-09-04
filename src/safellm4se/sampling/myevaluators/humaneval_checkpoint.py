"""Checkpoint persistence for interrupted HumanEval benchmark executions."""

import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

CHECKPOINT_DIRECTORY_NAME: str = "humaneval_checkpoints"  # Output subdirectory name.
CHECKPOINT_FILE_SUFFIX: str = ".json"  # Extension used for checkpoint files.
EXECUTION_FILE_PART_PREFIX: str = "execution-"  # Prefix for execution identifiers.
REQUIRED_RESULT_FIELDS: set[str] = {
    "completion",
    "completion_tokens",
    "entry_point",
    "passed",
    "problem_number",
    "prompt_tokens",
    "raw_text",
    "result",
    "task_id",
}  # Fields required to safely resume one completed program result.


def checkpoint_path_from_context(
    context: dict[str, Any],
    experiment_name: str,
    model_id: str,
) -> Path | None:
    """Build the checkpoint path for one sampler execution.

    Args:
        context: Runtime context values supplied by the sampler.
        experiment_name: Evaluator experiment name.
        model_id: Provider-specific model identifier.

    Returns:
        The checkpoint file path, or None when required context is absent.
    """
    output_dir_value: Any = context.get("output_dir")
    execution_number_value: Any = context.get("execution_number")
    task_id_value: Any = context.get("task_id")
    if (
        output_dir_value is None
        or execution_number_value is None
        or task_id_value is None
    ):
        return None

    output_dir: Path = Path(output_dir_value)
    execution_number: int = int(execution_number_value)
    return checkpoint_path_from_parts(
        output_dir,
        str(task_id_value),
        experiment_name,
        model_id,
        execution_number,
    )


def checkpoint_path_from_parts(
    output_dir: Path,
    task_id: str,
    experiment_name: str,
    model_id: str,
    execution_number: int,
) -> Path:
    """Build the checkpoint path for one HumanEval sampler execution.

    Args:
        output_dir: Directory where sampler output files are stored.
        task_id: Sampling task identifier.
        experiment_name: Evaluator experiment name.
        model_id: Provider-specific model identifier.
        execution_number: Sampler execution number.

    Returns:
        The checkpoint file path for the requested execution.
    """
    file_stem: str = "__".join(
        [
            _safe_path_part(task_id),
            _safe_path_part(experiment_name),
            _safe_path_part(model_id),
            f"{EXECUTION_FILE_PART_PREFIX}{execution_number}",
        ]
    )
    return (
        output_dir
        / CHECKPOINT_DIRECTORY_NAME
        / f"{file_stem}{CHECKPOINT_FILE_SUFFIX}"
    )


def list_checkpoint_execution_numbers(
    output_dir: Path,
    task_id: str,
    experiment_name: str,
    model_id: str,
) -> list[int]:
    """List unfinished checkpoint execution numbers for one sampling key.

    Args:
        output_dir: Directory where sampler output files are stored.
        task_id: Sampling task identifier.
        experiment_name: Evaluator experiment name.
        model_id: Provider-specific model identifier.

    Returns:
        Checkpoint execution numbers ordered by highest completed-result count.
    """
    checkpoint_directory: Path = output_dir / CHECKPOINT_DIRECTORY_NAME
    if not checkpoint_directory.exists():
        return []

    file_prefix: str = "__".join(
        [
            _safe_path_part(task_id),
            _safe_path_part(experiment_name),
            _safe_path_part(model_id),
            EXECUTION_FILE_PART_PREFIX,
        ]
    )
    ranked_execution_numbers: list[tuple[int, int]] = []
    for checkpoint_path in checkpoint_directory.glob(
        f"{file_prefix}*{CHECKPOINT_FILE_SUFFIX}"
    ):
        execution_number: int | None = _execution_number_from_checkpoint_path(
            checkpoint_path,
        )
        if execution_number is not None:
            completed_result_count: int = len(load_completed_results(checkpoint_path))
            ranked_execution_numbers.append(
                (completed_result_count, execution_number)
            )
    return [
        execution_number
        for _, execution_number in sorted(
            set(ranked_execution_numbers),
            key=lambda item: (-item[0], item[1]),
        )
    ]


def load_completed_results(checkpoint_path: Path | None) -> list[dict[str, Any]]:
    """Load completed HumanEval problem results from a checkpoint.

    Args:
        checkpoint_path: Checkpoint file path to read.

    Returns:
        Completed result dictionaries, or an empty list when no checkpoint exists.
    """
    if checkpoint_path is None:
        return []
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        return []

    try:
        checkpoint_data: Any = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exception:
        logger.warning(
            "Ignoring unreadable HumanEval checkpoint {}: {}.",
            checkpoint_path,
            exception,
        )
        return []

    raw_results: Any = checkpoint_data.get("completed_results")
    if not isinstance(raw_results, list):
        return []
    return [
        dict(result)
        for result in raw_results
        if _is_valid_completed_result(result)
    ]


def save_completed_results(
    checkpoint_path: Path | None,
    completed_results: list[dict[str, Any]],
) -> None:
    """Persist completed HumanEval problem results atomically.

    Args:
        checkpoint_path: Checkpoint file path to write.
        completed_results: Completed result dictionaries to persist.

    Raises:
        OSError: If the checkpoint file cannot be written.
    """
    if checkpoint_path is None:
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data: dict[str, Any] = {
        "completed_results": sorted(
            completed_results,
            key=lambda result: int(result["problem_number"]),
        ),
    }
    temporary_path: Path = checkpoint_path.with_suffix(
        f"{checkpoint_path.suffix}.tmp"
    )
    with temporary_path.open("w", encoding="utf-8") as checkpoint_file:
        json.dump(checkpoint_data, checkpoint_file, indent=2, sort_keys=True)
        checkpoint_file.write("\n")
        checkpoint_file.flush()
        os.fsync(checkpoint_file.fileno())
    temporary_path.replace(checkpoint_path)


def remove_checkpoint(checkpoint_path: str | Path | None) -> None:
    """Remove a HumanEval checkpoint after its final CSV row has been stored.

    Args:
        checkpoint_path: Checkpoint file path to remove.
    """
    if checkpoint_path is None:
        return

    path: Path = Path(checkpoint_path)
    try:
        path.unlink(missing_ok=True)
    except OSError as exception:
        logger.warning(
            "Could not remove HumanEval checkpoint {}: {}.",
            path,
            exception,
        )


def _safe_path_part(value: str) -> str:
    """Return a filesystem-safe identifier part.

    Args:
        value: Raw identifier value.

    Returns:
        A sanitized identifier suitable for a filename.
    """
    sanitized: str = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return sanitized or "unknown"


def _execution_number_from_checkpoint_path(checkpoint_path: Path) -> int | None:
    """Extract the execution number from a checkpoint file path.

    Args:
        checkpoint_path: Checkpoint path to inspect.

    Returns:
        The execution number, or None when the file name does not match.
    """
    stem: str = checkpoint_path.name.removesuffix(CHECKPOINT_FILE_SUFFIX)
    execution_part: str = stem.rsplit("__", 1)[-1]
    if not execution_part.startswith(EXECUTION_FILE_PART_PREFIX):
        return None
    raw_execution_number: str = execution_part.removeprefix(
        EXECUTION_FILE_PART_PREFIX
    )
    if not raw_execution_number.isdigit():
        return None
    return int(raw_execution_number)


def _is_valid_completed_result(result: Any) -> bool:
    """Return whether a checkpoint entry can be reused safely.

    Args:
        result: Raw checkpoint entry to inspect.

    Returns:
        True when the entry contains every required completed-result field.
    """
    if not isinstance(result, dict):
        return False
    if not REQUIRED_RESULT_FIELDS.issubset(result):
        return False
    return isinstance(result.get("problem_number"), int)
