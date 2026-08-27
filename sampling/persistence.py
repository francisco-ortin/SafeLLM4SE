"""Process-safe CSV persistence for measurements and summaries."""

import csv
import json
import math
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sampling.locking import interprocess_file_lock
from sampling.models import SamplerSettings, SamplingObservation
from sampling.statistics import confidence_interval

SAFE_MEASUREMENT_FIELDS: list[str] = [
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
MEASUREMENT_FIELDS: list[str] = SAFE_MEASUREMENT_FIELDS + [
    "value",
    "metric_type",
    "evaluator",
    "run_id",
    "metadata",
]
SUMMARY_FIELDS: list[str] = [
    "task_id",
    "model",
    "temperature",
    "metric_type",
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
    "ci_method",
    "target_ci_width",
    "budget",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


def row_matches(row: dict[str, Any], settings: SamplerSettings) -> bool:
    return (
        str(row.get("task_id")) == settings.task_id
        and str(row.get("model")) == settings.model
        and str(row.get("temperature")) == str(settings.temperature)
    )


def read_current_values(settings: SamplerSettings) -> list[float]:
    with interprocess_file_lock(settings.lock_path):
        rows = read_measurements(settings.measurements_path)
    return [
        float(row["value"])
        for row in rows
        if row_matches(row, settings) and str(row.get("value", "")) != ""
    ]


def read_measurements(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return [_normalize_measurement(row) for row in csv.DictReader(csv_file)]


def reserve_execution_number(settings: SamplerSettings) -> int | None:
    """Reserve the next sample index so parallel processes do not duplicate work."""

    with interprocess_file_lock(settings.lock_path):
        now = time.time()
        reservations = [
            item
            for item in _load_reservations(settings.reservations_path)
            if now - float(item.get("created_at", 0.0))
            <= settings.reservation_ttl_seconds
        ]
        measurements = read_measurements(settings.measurements_path)
        used = {
            int(row["execution_number"])
            for row in measurements
            if row_matches(row, settings)
            and str(row.get("execution_number", "")).isdigit()
        }
        for item in reservations:
            if tuple(item.get("key", [])) == _matching_key(settings):
                used.add(int(item["execution_number"]))

        for execution_number in range(1, settings.budget + 1):
            if execution_number not in used:
                reservations.append(
                    {
                        "key": list(_matching_key(settings)),
                        "execution_number": execution_number,
                        "run_id": settings.run_id,
                        "created_at": now,
                    }
                )
                _write_reservations(settings.reservations_path, reservations)
                return execution_number
        _write_reservations(settings.reservations_path, reservations)
        return None


def append_measurement_process_safe(
    settings: SamplerSettings,
    row: dict[str, Any],
) -> None:
    with interprocess_file_lock(settings.lock_path):
        rows = read_measurements(settings.measurements_path)
        keys = {_identity(existing) for existing in rows}
        normalized = _normalize_measurement(row)
        if _identity(normalized) not in keys:
            rows.append(normalized)
            _write_measurements(settings.measurements_path, rows)
        remove_reservation(settings, int(normalized["execution_number"]))


def remove_reservation(settings: SamplerSettings, execution_number: int) -> None:
    reservations = [
        item
        for item in _load_reservations(settings.reservations_path)
        if not (
            tuple(item.get("key", [])) == _matching_key(settings)
            and int(item.get("execution_number", -1)) == execution_number
            and item.get("run_id") == settings.run_id
        )
    ]
    _write_reservations(settings.reservations_path, reservations)


def create_measurement_row(
    settings: SamplerSettings,
    execution_number: int,
    observation: SamplingObservation,
) -> dict[str, Any]:
    timestamp = datetime.now()
    return {
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "task_id": settings.task_id,
        "model": settings.model,
        "temperature": settings.temperature,
        "execution_number": execution_number,
        "prompt_tokens": observation.prompt_tokens,
        "completion_tokens": observation.completion_tokens,
        "total_tokens": observation.total_tokens,
        "value": observation.theta,
        "metric_type": settings.metric_type,
        "evaluator": settings.evaluator_name,
        "run_id": settings.run_id,
        "metadata": json.dumps(observation.metadata, sort_keys=True),
    }


def summarize(settings: SamplerSettings) -> list[dict[str, Any]]:
    with interprocess_file_lock(settings.lock_path):
        rows = read_measurements(settings.measurements_path)

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("task_id")),
            str(row.get("model")),
            str(row.get("temperature")),
            str(row.get("metric_type") or "binary"),
        )
        groups.setdefault(key, []).append(row)

    summary_rows = [_build_summary_row(settings, key, rows) for key, rows in groups.items()]
    summary_rows = [row for row in summary_rows if row]
    with interprocess_file_lock(settings.lock_path):
        settings.results_path.parent.mkdir(parents=True, exist_ok=True)
        with settings.results_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)
            csv_file.flush()
            os.fsync(csv_file.fileno())
    return summary_rows


def _build_summary_row(
    settings: SamplerSettings,
    key: tuple[str, str, str, str],
    group_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    task_id, model, temperature, metric_type = key
    values = [
        float(row["value"])
        for row in group_rows
        if str(row.get("value", "")) != ""
    ]
    if not values:
        return {}

    n = len(values)
    mean_value = statistics.fmean(values)
    std_value = statistics.stdev(values) if n > 1 else 0.0
    ci_low, ci_high, ci_method = confidence_interval(
        values,
        metric_type,
        settings.confidence_level,
        settings.ci_method,
        settings.bootstrap_samples,
    )
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in group_rows)
    completion_tokens = sum(
        int(row.get("completion_tokens") or 0) for row in group_rows
    )
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in group_rows)
    successes = sum(1 for value in values if int(round(value)) == 1)
    error_margin = (ci_high - ci_low) / 2.0
    return {
        "task_id": task_id,
        "model": model,
        "temperature": temperature,
        "metric_type": metric_type,
        "n": n,
        "mean": mean_value,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": std_value,
        "cov": std_value / mean_value if mean_value != 0 else math.nan,
        "success_rate": successes / n if metric_type == "binary" else "",
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "error_margin_absolute": error_margin,
        "error_margin_relative": (
            error_margin / mean_value if mean_value != 0 else math.nan
        ),
        "ci_method": ci_method,
        "target_ci_width": settings.target_ci_width,
        "budget": settings.budget,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("task_id", "")),
        str(row.get("model", "")),
        str(row.get("temperature", "")),
        str(row.get("execution_number", "")),
    )


def _normalize_measurement(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field, "") for field in MEASUREMENT_FIELDS}
    if not normalized["value"]:
        normalized["value"] = normalized.get("passed", "")
    return normalized


def _write_measurements(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        normalized = _normalize_measurement(row)
        unique[_identity(normalized)] = normalized
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(unique.values())
        csv_file.flush()
        os.fsync(csv_file.fileno())


def _matching_key(settings: SamplerSettings) -> tuple[str, str, str]:
    return settings.task_id, settings.model, str(settings.temperature)


def _load_reservations(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _write_reservations(path: Path, reservations: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reservations, indent=2), encoding="utf-8")
