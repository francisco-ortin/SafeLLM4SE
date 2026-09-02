"""CSV input loading and validation for SafeLLM4SE sampling reports."""

import csv
from pathlib import Path
from typing import Any

REQUIRED_INPUT_FIELDS: tuple[str, ...] = (
    "task_id",
    "model_name",
    "model_id",
    "temperature",
    "theta",
    "prompt_tokens",
    "completion_tokens",
)
CONSISTENCY_FIELDS: tuple[str, ...] = ("model_name", "model_id", "temperature")


def read_task_rows(input_path: Path, task_id: str) -> list[dict[str, str]]:
    """Read sampling rows matching a task identifier.

    Args:
        input_path: Sampling CSV path to read.
        task_id: Task identifier used to filter rows.

    Returns:
        Rows whose task_id column equals the requested task identifier.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        ValueError: If the CSV is empty, lacks required columns, or has no rows
            for the requested task.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")
    if input_path.stat().st_size == 0:
        raise ValueError(f"Input CSV file is empty: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader: csv.DictReader[str] = csv.DictReader(csv_file)
        _validate_required_fields(reader.fieldnames)
        rows: list[dict[str, str]] = [
            _normalize_row(row)
            for row in reader
            if str(row.get("task_id", "")) == task_id
        ]
    if not rows:
        raise ValueError(f"No rows found for task_id '{task_id}'.")
    validate_consistent_fields(rows, CONSISTENCY_FIELDS)
    return rows


def validate_consistent_fields(
    rows: list[dict[str, str]],
    fieldnames: tuple[str, ...],
) -> None:
    """Validate that all rows share the same values for selected fields.

    Args:
        rows: Input rows to validate.
        fieldnames: Field names that must be constant across all rows.

    Raises:
        ValueError: If any selected field contains more than one value.
    """
    for fieldname in fieldnames:
        values: set[str] = {row.get(fieldname, "") for row in rows}
        if len(values) > 1:
            formatted_values: str = ", ".join(sorted(repr(value) for value in values))
            raise ValueError(
                f"Inconsistent values for '{fieldname}' in task_id "
                f"'{rows[0].get('task_id', '')}': {formatted_values}."
            )


def _validate_required_fields(fieldnames: list[str] | None) -> None:
    """Validate that a CSV header includes all required fields.

    Args:
        fieldnames: CSV header field names.

    Raises:
        ValueError: If the CSV header is missing or required fields are absent.
    """
    if fieldnames is None:
        raise ValueError("Input CSV file does not contain a header row.")
    missing_fields: list[str] = [
        fieldname for fieldname in REQUIRED_INPUT_FIELDS if fieldname not in fieldnames
    ]
    if missing_fields:
        fields_text: str = ", ".join(missing_fields)
        raise ValueError(f"Input CSV file is missing required fields: {fields_text}.")


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    """Convert CSV row values to strings.

    Args:
        row: Raw CSV row.

    Returns:
        A row with string values and empty strings for missing values.
    """
    return {
        str(fieldname): "" if value is None else str(value)
        for fieldname, value in row.items()
    }
