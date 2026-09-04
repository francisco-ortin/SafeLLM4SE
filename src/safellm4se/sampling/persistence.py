"""Process-safe CSV persistence for sampler measurements."""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from safellm4se.sampling.locking import interprocess_file_lock
from safellm4se.sampling.models import SamplerSettings, SamplingObservation

SAFE_MEASUREMENT_FIELDS: list[str] = [
    "date",
    "time",
    "task_id",
    "experiment_name",
    "model_name",
    "model_id",
    "execution_number",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]
MEASUREMENT_FIELDS: list[str] = SAFE_MEASUREMENT_FIELDS + [
    "theta",
    "metric_type",
    "evaluator",
    "evaluator_parameters",
    "run_id",
]
RESERVED_MEASUREMENT_FIELDS: set[str] = set(MEASUREMENT_FIELDS)
REMOVED_MEASUREMENT_FIELDS: set[str] = {"metadata", "model", "passed", "value"}
NULL_CSV_VALUE: str = ""


def row_matches(
    row: dict[str, Any],
    settings: SamplerSettings,
    experiment_name: str,
    model_id: str,
) -> bool:
    """Return whether a measurement row belongs to the current sampling key.
    Args:
        row: Measurement row to inspect.
        settings: Sampler settings containing the target task identifier.
        experiment_name: Expected experiment name.
        model_id: Expected model identifier.
    Returns:
        True if the row belongs to the task, experiment, and model id;
        otherwise, False.
    """
    return (
        str(row.get("task_id")) == settings.task_id
        and str(row.get("experiment_name", "")) == experiment_name
        and _row_model_id(row) == model_id
    )


def read_current_theta(
    settings: SamplerSettings,
    experiment_name: str,
    model_id: str,
) -> list[float]:
    """Read persisted theta observations for the current sampling key.
    Args:
        settings: Sampler settings containing paths and task identifier.
        experiment_name: Experiment name used to filter rows.
        model_id: Model identifier used to filter rows.
    Returns:
        Persisted theta observations for the task, experiment, and model id.
    """
    theta_values, _ = read_current_theta_and_total_tokens(
        settings,
        experiment_name,
        model_id,
    )
    return theta_values


def read_current_theta_and_total_tokens(
    settings: SamplerSettings,
    experiment_name: str,
    model_id: str,
) -> tuple[list[float], int]:
    """Read persisted theta observations and tokens for the current sampling key.
    Args:
        settings: Sampler settings containing paths and task identifier.
        experiment_name: Experiment name used to filter rows.
        model_id: Model identifier used to filter rows.
    Returns:
        A tuple containing theta observations and the summed token count.
    """
    with interprocess_file_lock(settings.lock_path):
        rows = read_measurements(settings.measurements_path)
    matching_rows: list[dict[str, Any]] = [
        row
        for row in rows
        if row_matches(row, settings, experiment_name, model_id)
    ]
    theta_values: list[float] = [
        float(row["theta"])
        for row in matching_rows
        if str(row.get("theta", "")) != ""
    ]
    total_tokens: int = sum(
        int(row.get("total_tokens", 0) or 0)
        for row in matching_rows
        if str(row.get("total_tokens", "")).isdigit()
    )
    return theta_values, total_tokens


def read_measurements(path: Path) -> list[dict[str, Any]]:
    """Read measurement rows from a CSV file.
    Args:
        path: Measurements CSV path.
    Returns:
        Normalized measurement rows, or an empty list when the file is absent or
        empty.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return [_normalize_measurement(row) for row in csv.DictReader(csv_file)]


def reserve_execution_number(
    settings: SamplerSettings,
    experiment_name: str,
    model_id: str,
    reusable_execution_numbers: list[int] | None = None,
) -> int:
    """Reserve the next sample index so parallel processes do not duplicate work.
    Args:
        settings: Sampler settings containing persistence paths and run identity.
        experiment_name: Experiment name used in the reservation key.
        model_id: Model identifier used in the reservation key.
        reusable_execution_numbers: Incomplete execution numbers that can be
            resumed because they have persisted partial results.
    Returns:
        The reserved execution number.
    """
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
            if row_matches(row, settings, experiment_name, model_id)
            and str(row.get("execution_number", "")).isdigit()
        }
        reusable: list[int] = [
            execution_number
            for execution_number in reusable_execution_numbers or []
            if execution_number not in used
        ]
        for execution_number in reusable:
            matching_reservation: dict[str, Any] | None = next(
                (
                    item
                    for item in reservations
                    if tuple(item.get("key", []))
                    == _matching_key(settings, experiment_name, model_id)
                    and int(item.get("execution_number", -1)) == execution_number
                ),
                None,
            )
            if matching_reservation is None:
                reservations.append(
                    {
                        "key": list(
                            _matching_key(settings, experiment_name, model_id)
                        ),
                        "execution_number": execution_number,
                        "run_id": settings.run_id,
                        "created_at": now,
                    }
                )
            else:
                matching_reservation["run_id"] = settings.run_id
                matching_reservation["created_at"] = now
            _write_reservations(settings.reservations_path, reservations)
            return execution_number

        for item in reservations:
            if tuple(item.get("key", [])) == _matching_key(
                settings,
                experiment_name,
                model_id,
            ):
                used.add(int(item["execution_number"]))

        execution_number: int = max(used, default=0) + 1
        reservations.append(
            {
                "key": list(_matching_key(settings, experiment_name, model_id)),
                "execution_number": execution_number,
                "run_id": settings.run_id,
                "created_at": now,
            }
        )
        _write_reservations(settings.reservations_path, reservations)
        return execution_number


def append_measurement_process_safe(
    settings: SamplerSettings,
    row: dict[str, Any],
    experiment_name: str,
    model_id: str,
) -> None:
    """Append a measurement row while holding the interprocess lock.
    Args:
        settings: Sampler settings containing persistence paths and run identity.
        row: Measurement row to append.
        experiment_name: Experiment name used to remove the matching reservation.
        model_id: Model identifier used to remove the matching reservation.
    """
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
            experiment_name,
            model_id,
        )


def remove_reservation(
    settings: SamplerSettings,
    execution_number: int,
    experiment_name: str,
    model_id: str,
) -> None:
    """Remove one execution reservation for the current sampling key.
    Args:
        settings: Sampler settings containing reservation path and run identity.
        execution_number: Reserved execution number to remove.
        experiment_name: Experiment name used in the reservation key.
        model_id: Model identifier used in the reservation key.
    """
    reservations = [
        item
        for item in _load_reservations(settings.reservations_path)
        if not (
            tuple(item.get("key", []))
            == _matching_key(settings, experiment_name, model_id)
            and int(item.get("execution_number", -1)) == execution_number
            and item.get("run_id") == settings.run_id
        )
    ]
    _write_reservations(settings.reservations_path, reservations)


def create_measurement_row(
    settings: SamplerSettings,
    execution_number: int,
    observation: SamplingObservation,
    evaluator_name: str,
    metric_type: str,
) -> dict[str, Any]:
    """Create one CSV-ready measurement row from a sampling observation.
    Args:
        settings: Sampler settings containing task and run metadata.
        execution_number: Execution number assigned to the observation.
        observation: Sampling observation to serialize.
        evaluator_name: Name of the evaluator class that produced the observation.
        metric_type: Metric type associated with the observation.
    Returns:
        A CSV-ready measurement row.
    """
    timestamp = datetime.now()
    evaluator_parameters: dict[str, Any] = settings.evaluator_parameters
    row: dict[str, Any] = {
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "task_id": settings.task_id,
        "experiment_name": observation.experiment_name,
        "model_name": observation.model_name,
        "model_id": observation.model_id,
        "execution_number": execution_number,
        "prompt_tokens": observation.prompt_tokens,
        "completion_tokens": observation.completion_tokens,
        "total_tokens": observation.total_tokens,
        "theta": observation.theta,
        "metric_type": metric_type,
        "evaluator": evaluator_name,
        "evaluator_parameters": json.dumps(
            evaluator_parameters,
            sort_keys=True,
            default=str,
        ),
        "run_id": settings.run_id,
    }
    row.update(_free_parameter_fields(evaluator_parameters))
    return row


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return the identity tuple used to deduplicate measurement rows.
    Args:
        row: Measurement row to identify.
    Returns:
        A tuple of task identifier, experiment name, model identifier, and
        execution number.
    """
    return (
        str(row.get("task_id", "")),
        str(row.get("experiment_name", "")),
        _row_model_id(row),
        str(row.get("execution_number", "")),
    )


def _normalize_measurement(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw CSV row while preserving dynamic parameter fields.
    Args:
        row: Raw CSV measurement row.
    Returns:
        A normalized row with core fields and preserved dynamic parameter fields.
    """
    normalized = {field: row.get(field, "") for field in MEASUREMENT_FIELDS}
    if normalized["theta"] is None or str(normalized["theta"]) == "":
        normalized["theta"] = str(row.get("value", ""))
    if not normalized["model_name"]:
        normalized["model_name"] = _row_model_name(row)
    normalized.update(_legacy_free_parameter_fields(row))
    return normalized


def _write_measurements(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write measurement rows to disk with stable core and dynamic fields.
    Args:
        path: Destination measurements CSV path.
        rows: Measurement rows to persist.
    """
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
    experiment_name: str,
    model_id: str,
) -> tuple[str, str, str]:
    """Return the reservation key for one task, experiment, and model id.
    Args:
        settings: Sampler settings containing the task identifier.
        experiment_name: Experiment name for the reservation.
        model_id: Model identifier for the reservation.
    Returns:
        A reservation key tuple.
    """
    return settings.task_id, experiment_name, model_id


def _row_model_name(row: dict[str, Any]) -> str:
    """Return the canonical model name from a new or legacy measurement row.
    Args:
        row: Measurement row to inspect.
    Returns:
        The model name, or an empty string if absent.
    """
    return str(row.get("model_name") or row.get("model") or "")


def _row_model_id(row: dict[str, Any]) -> str:
    """Return the model id from a measurement row.
    Args:
        row: Measurement row to inspect.
    Returns:
        The model identifier, or an empty string if absent.
    """
    return str(row.get("model_id") or "")


def _serialize_csv_value(value: Any) -> str:
    """Serialize a user-provided evaluator parameter for a CSV cell.
    Args:
        value: Evaluator parameter value to serialize.
    Returns:
        A string suitable for a CSV cell.
    """
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _free_parameter_fields(evaluator_parameters: dict[str, Any]) -> dict[str, str]:
    """Return CSV fields for non-reserved evaluator parameters.
    Args:
        evaluator_parameters: Evaluator parameters to expose as dynamic fields.
    Returns:
        Non-reserved evaluator parameters serialized for CSV output.
    """
    return {
        name: _serialize_csv_value(value)
        for name, value in evaluator_parameters.items()
        if name not in RESERVED_MEASUREMENT_FIELDS
        and name not in REMOVED_MEASUREMENT_FIELDS
    }


def _legacy_free_parameter_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return non-reserved fields preserved from existing measurement rows.
    Args:
        row: Existing measurement row.
    Returns:
        Non-reserved fields preserved from the row.
    """
    return {
        field: value
        for field, value in row.items()
        if isinstance(field, str)
        and field not in RESERVED_MEASUREMENT_FIELDS
        and field not in REMOVED_MEASUREMENT_FIELDS
    }


def _measurement_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Return stable measurement fieldnames including dynamic parameter columns.
    Args:
        rows: Measurement rows to inspect.
    Returns:
        Ordered CSV field names including dynamic parameter columns.
    """
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
    """Return rows with explicit null values for missing CSV fields.
    Args:
        rows: Measurement rows to normalize.
        fieldnames: Field names that each output row must contain.
    Returns:
        Rows with explicit null values for missing fields.
    """
    return [
        {
            fieldname: row.get(fieldname, NULL_CSV_VALUE)
            for fieldname in fieldnames
        }
        for row in rows
    ]


def _load_reservations(path: Path) -> list[dict[str, Any]]:
    """Load current execution reservations from disk.
    Args:
        path: Reservation JSON path.
    Returns:
        Current reservations, or an empty list when the file is absent, empty, or
        contains invalid JSON.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _write_reservations(path: Path, reservations: list[dict[str, Any]]) -> None:
    """Persist execution reservations to disk.
    Args:
        path: Reservation JSON path.
        reservations: Reservations to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reservations, indent=2), encoding="utf-8")
