"""CSV persistence and aggregation helpers for SAFE measurements."""

import csv
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from sampling.config.config import Config

MEASUREMENT_FIELDS: list[str] = [
    "date",
    "time",
    "task_id",
    "model",
    "temperature",
    "execution_number",
    "passed",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]
SUMMARY_FIELDS: list[str] = [
    "task_id",
    "model",
    "temperature",
    "n",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "cov",
    "success_rate",
    "ci_95_low",
    "ci_95_high",
    "error_margin_absolute",
    "error_margin_relative",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


def get_results_file_paths(log_file: Optional[str] = None) -> tuple[str, str]:
    """Return the configured measurement and summary CSV output paths."""
    del log_file
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    measurements_path: str = os.path.join(
        Config.RESULTS_DIR,
        Config.MEASUREMENTS_FILE_NAME,
    )
    results_path: str = os.path.join(Config.RESULTS_DIR, Config.RESULTS_FILE_NAME)
    return measurements_path, results_path


def _current_model_key() -> tuple[str, str]:
    """Return the configured model and temperature as a grouping key."""
    return Config.MODEL, str(Config.TEMPERATURE)


def _normalize_measurement_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a CSV measurement row to the configured output schema."""
    normalized: dict[str, Any] = {
        field: row.get(field, "")
        for field in MEASUREMENT_FIELDS
    }
    normalized["model"] = normalized["model"] or Config.MODEL
    normalized["temperature"] = normalized["temperature"] or str(Config.TEMPERATURE)
    return normalized


def _measurement_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return the unique identity fields for one measurement row."""
    normalized: dict[str, Any] = _normalize_measurement_row(row)
    return (
        str(normalized.get("task_id", "")),
        str(normalized.get("model", "")),
        str(normalized.get("temperature", "")),
        str(normalized.get("execution_number", "")),
    )


def _deduplicate_measurements(
    measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only one measurement per task, model, temperature, and execution."""
    unique_measurements: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in measurements:
        normalized = _normalize_measurement_row(row)
        unique_measurements[_measurement_identity(normalized)] = normalized
    return list(unique_measurements.values())


def read_measurements_csv(input_path: str) -> list[dict[str, Any]]:
    """Read measurements from a CSV file, returning an empty list if missing."""
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        return []

    with open(input_path, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [_normalize_measurement_row(row) for row in reader]


def get_completed_execution_numbers(
    measurements: list[dict[str, Any]],
    task_id: str,
    model_name: Optional[str] = None,
) -> set[int]:
    """Return execution numbers already measured for one task and model."""
    model: str = model_name or Config.MODEL
    temperature: str = str(Config.TEMPERATURE)
    completed: set[int] = set()
    for row in measurements:
        if (
            row.get("task_id") == task_id
            and row.get("model") == model
            and str(row.get("temperature")) == temperature
        ):
            try:
                completed.add(int(row["execution_number"]))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def create_measurement_row(
    task_id: str,
    execution_number: int,
    passed: int,
    token_usage: dict[str, int],
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """Create a normalized measurement row for one generated completion."""
    timestamp: datetime = datetime.now()
    return {
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "task_id": task_id,
        "model": model_name or Config.MODEL,
        "temperature": Config.TEMPERATURE,
        "execution_number": execution_number,
        "passed": passed,
        **token_usage,
    }


def write_measurements_csv(
    measurements: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write all measurement rows to a CSV file, replacing existing content."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(_deduplicate_measurements(measurements))
        csv_file.flush()
        os.fsync(csv_file.fileno())


def append_measurements_csv(
    measurements: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Append only new measurement rows to a CSV file."""
    if not measurements:
        if not os.path.exists(output_path):
            write_measurements_csv([], output_path)
        return

    existing_measurements: list[dict[str, Any]] = read_measurements_csv(output_path)
    existing_keys: set[tuple[str, str, str, str]] = {
        _measurement_identity(row)
        for row in existing_measurements
    }
    should_write_header: bool = (
        not os.path.exists(output_path)
        or os.path.getsize(output_path) == 0
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEASUREMENT_FIELDS)
        if should_write_header:
            writer.writeheader()

        for row in measurements:
            normalized = _normalize_measurement_row(row)
            key = _measurement_identity(normalized)
            if key in existing_keys:
                continue
            writer.writerow(normalized)
            existing_keys.add(key)

        csv_file.flush()
        os.fsync(csv_file.fileno())


def build_raw_results_from_measurements(
    measurements: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[int]]:
    """Group pass or fail results by task, model, and temperature."""
    raw_results: dict[tuple[str, str, str], list[int]] = {}
    for row in _deduplicate_measurements(measurements):
        key = (
            str(row.get("task_id", "")),
            str(row.get("model") or Config.MODEL),
            str(row.get("temperature") or Config.TEMPERATURE),
        )
        try:
            passed = int(row.get("passed", 0))
        except (TypeError, ValueError):
            passed = 0
        raw_results.setdefault(key, []).append(passed)
    return raw_results


def write_summary_csv(
    measurements: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write aggregate SAFE statistics from raw measurements to a CSV file."""
    token_totals: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in _deduplicate_measurements(measurements):
        key = (
            str(row.get("task_id", "")),
            str(row.get("model") or Config.MODEL),
            str(row.get("temperature") or Config.TEMPERATURE),
        )
        totals: dict[str, int] = token_totals.setdefault(
            key,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        totals["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(row.get("completion_tokens") or 0)
        totals["total_tokens"] += int(row.get("total_tokens") or 0)

    summary_rows: list[dict[str, Any]] = []
    raw_results: dict[tuple[str, str, str], list[int]] = (
        build_raw_results_from_measurements(measurements)
    )
    for (task_id, model, temperature), passes in raw_results.items():
        n: int = len(passes)
        if n == 0:
            continue

        successes: int = sum(passes)
        mean_value: float = float(np.mean(passes)) if n > 0 else float("nan")
        std_value: float = float(np.std(passes, ddof=1)) if n > 1 else 0.0
        ci_low, ci_high = (
            proportion_confint(successes, n, method="wilson", alpha=0.05)
            if n > 0
            else (float("nan"), float("nan"))
        )
        error_margin_absolute: float = (
            float((ci_high - ci_low) / 2)
            if n > 0
            else float("nan")
        )
        token_values: dict[str, int] = token_totals.get(
            (task_id, model, temperature),
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

        summary_rows.append(
            {
                "task_id": task_id,
                "model": model,
                "temperature": temperature,
                "n": n,
                "mean": mean_value,
                "median": float(np.median(passes)) if n > 0 else float("nan"),
                "min": int(np.min(passes)) if n > 0 else float("nan"),
                "max": int(np.max(passes)) if n > 0 else float("nan"),
                "std": std_value,
                "cov": (
                    std_value / mean_value
                    if mean_value != 0
                    else float("nan")
                ),
                "success_rate": successes / n if n > 0 else float("nan"),
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                "error_margin_absolute": error_margin_absolute,
                "error_margin_relative": (
                    error_margin_absolute / mean_value
                    if mean_value != 0
                    else float("nan")
                ),
                "prompt_tokens": token_values["prompt_tokens"],
                "completion_tokens": token_values["completion_tokens"],
                "total_tokens": token_values["total_tokens"],
            }
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
        csv_file.flush()
        os.fsync(csv_file.fileno())
