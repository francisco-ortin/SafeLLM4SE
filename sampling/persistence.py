"""Process-safe CSV persistence for sampler measurements."""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sampling.locking import interprocess_file_lock
from sampling.models import SamplerSettings, SamplingObservation

SAFE_MEASUREMENT_FIELDS: list[str] = [
    "date",
    "time",
    "task_id",
    "model_name",
    "model_id",
    "execution_number",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]
MEASUREMENT_FIELDS: list[str] = SAFE_MEASUREMENT_FIELDS + [
    "value",
    "metric_type",
    "evaluator",
    "evaluator_parameters",
    "run_id",
    "metadata",
]
RESERVED_MEASUREMENT_FIELDS: set[str] = set(MEASUREMENT_FIELDS)
REMOVED_MEASUREMENT_FIELDS: set[str] = {"model", "passed"}
NULL_CSV_VALUE: str = ""


def row_matches(
    row: dict[str, Any],
    settings: SamplerSettings,
    model_name: str,
    model_id: str,
) -> bool:
    """Return whether a measurement row belongs to the current task and model."""

    return (
        str(row.get("task_id")) == settings.task_id
        and _row_model_name(row) == model_name
        and _row_model_id(row) in {"", model_id}
    )


def read_current_values(
    settings: SamplerSettings,
    model_name: str,
    model_id: str,
) -> list[float]:
    """Read persisted metric values for the current task and evaluator model."""

    with interprocess_file_lock(settings.lock_path):
        rows = read_measurements(settings.measurements_path)
    return [
        float(row["value"])
        for row in rows
        if row_matches(row, settings, model_name, model_id)
        and str(row.get("value", "")) != ""
    ]


def read_measurements(path: Path) -> list[dict[str, Any]]:
    """Read measurement rows from a CSV file."""

    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return [_normalize_measurement(row) for row in csv.DictReader(csv_file)]


def reserve_execution_number(
    settings: SamplerSettings,
    model_name: str,
    model_id: str,
) -> int | None:
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
            if row_matches(row, settings, model_name, model_id)
            and str(row.get("execution_number", "")).isdigit()
        }
        for item in reservations:
            if tuple(item.get("key", [])) == _matching_key(
                settings,
                model_name,
                model_id,
            ):
                used.add(int(item["execution_number"]))

        for execution_number in range(1, settings.budget + 1):
            if execution_number not in used:
                reservations.append(
                    {
                        "key": list(_matching_key(settings, model_name, model_id)),
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
    model_name: str,
    model_id: str,
) -> None:
    """Append a measurement row while holding the interprocess lock."""

    with interprocess_file_lock(settings.lock_path):
        rows = read_measurements(settings.measurements_path)
        keys = {_identity(existing) for existing in rows}
        normalized = _normalize_measurement(row)
        if _identity(normalized) not in keys:
            rows.append(normalized)
            _write_measurements(settings.measurements_path, rows)
        remove_reservation(
            settings,
            int(normalized["execution_number"]),
            model_name,
            model_id,
        )


def remove_reservation(
    settings: SamplerSettings,
    execution_number: int,
    model_name: str,
    model_id: str,
) -> None:
    """Remove one execution reservation for the current task and model."""

    reservations = [
        item
        for item in _load_reservations(settings.reservations_path)
        if not (
            tuple(item.get("key", []))
            == _matching_key(settings, model_name, model_id)
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
    """Create one CSV-ready measurement row from a sampling observation."""

    timestamp = datetime.now()
    evaluator_parameters: dict[str, Any] = settings.evaluator_parameters
    row: dict[str, Any] = {
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "task_id": settings.task_id,
        "model_name": observation.model_name,
        "model_id": observation.model_id,
        "execution_number": execution_number,
        "prompt_tokens": observation.prompt_tokens,
        "completion_tokens": observation.completion_tokens,
        "total_tokens": observation.total_tokens,
        "value": observation.theta,
        "metric_type": settings.metric_type,
        "evaluator": settings.evaluator_name,
        "evaluator_parameters": json.dumps(
            evaluator_parameters,
            sort_keys=True,
            default=str,
        ),
        "run_id": settings.run_id,
        "metadata": json.dumps(observation.metadata, sort_keys=True),
    }
    row.update(_free_parameter_fields(evaluator_parameters))
    return row


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return the identity tuple used to deduplicate measurement rows."""

    return (
        str(row.get("task_id", "")),
        _row_model_name(row),
        _row_model_id(row),
        str(row.get("execution_number", "")),
    )


def _normalize_measurement(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw CSV row while preserving dynamic parameter fields."""

    normalized = {field: row.get(field, "") for field in MEASUREMENT_FIELDS}
    if not normalized["model_name"]:
        normalized["model_name"] = str(normalized.get("model", ""))
    normalized.update(_legacy_free_parameter_fields(row))
    return normalized


def _write_measurements(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write measurement rows to disk with stable core and dynamic fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        normalized = _normalize_measurement(row)
        unique[_identity(normalized)] = normalized
    fieldnames: list[str] = _measurement_fieldnames(list(unique.values()))
    output_rows: list[dict[str, Any]] = _rows_with_null_fields(
        list(unique.values()),
        fieldnames,
    )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def _matching_key(
    settings: SamplerSettings,
    model_name: str,
    model_id: str,
) -> tuple[str, str, str]:
    """Return the reservation key for one task and model pair."""

    return settings.task_id, model_name, model_id


def _row_model_name(row: dict[str, Any]) -> str:
    """Return the canonical model name from a new or legacy measurement row."""

    return str(row.get("model_name") or row.get("model") or "")


def _row_model_id(row: dict[str, Any]) -> str:
    """Return the model id from a measurement row."""

    return str(row.get("model_id") or "")


def _serialize_csv_value(value: Any) -> str:
    """Serialize a user-provided evaluator parameter for a CSV cell."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _free_parameter_fields(evaluator_parameters: dict[str, Any]) -> dict[str, str]:
    """Return CSV fields for non-reserved evaluator parameters."""

    return {
        name: _serialize_csv_value(value)
        for name, value in evaluator_parameters.items()
        if name not in RESERVED_MEASUREMENT_FIELDS
        and name not in REMOVED_MEASUREMENT_FIELDS
    }


def _legacy_free_parameter_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return non-reserved fields preserved from existing measurement rows."""

    return {
        field: value
        for field, value in row.items()
        if isinstance(field, str)
        and field not in RESERVED_MEASUREMENT_FIELDS
        and field not in REMOVED_MEASUREMENT_FIELDS
    }


def _measurement_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Return stable measurement fieldnames including dynamic parameter columns."""

    dynamic_fieldnames: list[str] = sorted(
        {
            fieldname
            for row in rows
            for fieldname in row
            if fieldname not in RESERVED_MEASUREMENT_FIELDS
            and fieldname not in REMOVED_MEASUREMENT_FIELDS
        }
    )
    return MEASUREMENT_FIELDS + [
        fieldname
        for fieldname in dynamic_fieldnames
        if fieldname not in MEASUREMENT_FIELDS
    ]


def _rows_with_null_fields(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> list[dict[str, Any]]:
    """Return rows with explicit null values for missing CSV fields."""

    return [
        {
            fieldname: row.get(fieldname, NULL_CSV_VALUE)
            for fieldname in fieldnames
        }
        for row in rows
    ]


def _load_reservations(path: Path) -> list[dict[str, Any]]:
    """Load current execution reservations from disk."""

    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _write_reservations(path: Path, reservations: list[dict[str, Any]]) -> None:
    """Persist execution reservations to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reservations, indent=2), encoding="utf-8")
