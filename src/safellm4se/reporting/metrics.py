"""Report metric calculations for SafeLLM4SE sampling measurements."""

import math
import statistics
from datetime import datetime
from typing import Any

from safellm4se.sampling.config import config
from safellm4se.statistical_utils import confidence_interval, validate_confidence_level

REPORT_FIELDS: list[str] = [
    "date",
    "time",
    "task_id",
    "model_name",
    "model_id",
    "temperature",
    "N",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "theta_mean",
    "theta_median",
    "theta_min",
    "theta_max",
    "theta_type",
    "sd",
    "cv",
    "iqr",
    "q1",
    "q3",
    "ci_method",
    "ci_confidence-level",
    "ci_low",
    "ci_high",
    "ci_width",
]


def build_report_row(
    rows: list[dict[str, str]],
    confidence_level: float,
    ci_method: str,
) -> dict[str, Any]:
    """Build one report row from filtered sampling measurements.

    Args:
        rows: Sampling rows for one task.
        confidence_level: Confidence level used for confidence intervals.
        ci_method: Requested confidence interval method for continuous metrics.

    Returns:
        A CSV-ready report row with quality, variability, and uncertainty data.

    Raises:
        ValueError: If the confidence level is invalid or required numeric fields
            cannot be parsed.
    """
    validate_confidence_level(confidence_level)
    theta_values: list[float] = parse_theta_values(rows)
    theta_type: str = infer_theta_type(rows, theta_values)
    ci_low, ci_high, selected_ci_method = confidence_interval(
        theta_values,
        theta_type,
        confidence_level,
        ci_method,
        config.bootstrap_samples,
    )
    q1_value: float = percentile(theta_values, 0.25)
    q3_value: float = percentile(theta_values, 0.75)
    mean_value: float = statistics.fmean(theta_values)
    standard_deviation: float = (
        statistics.stdev(theta_values) if len(theta_values) > 1 else 0.0
    )
    prompt_tokens: int = integer_column_sum(rows, "prompt_tokens")
    completion_tokens: int = integer_column_sum(rows, "completion_tokens")
    timestamp: datetime = datetime.now()
    return {
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "task_id": rows[0]["task_id"],
        "model_name": rows[0]["model_name"],
        "model_id": rows[0]["model_id"],
        "temperature": rows[0]["temperature"],
        "N": len(rows),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "theta_mean": mean_value,
        "theta_median": statistics.median(theta_values),
        "theta_min": min(theta_values),
        "theta_max": max(theta_values),
        "theta_type": theta_type,
        "sd": standard_deviation,
        "cv": coefficient_of_variation(mean_value, standard_deviation),
        "iqr": q3_value - q1_value,
        "q1": q1_value,
        "q3": q3_value,
        "ci_method": selected_ci_method,
        "ci_confidence-level": confidence_level * 100.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
    }


def percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile.

    Args:
        values: Numeric sample values.
        probability: Percentile probability in the closed interval [0, 1].

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If values is empty or probability is outside [0, 1].
    """
    if not values:
        raise ValueError("values cannot be empty.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")
    sorted_values: list[float] = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position: float = probability * (len(sorted_values) - 1)
    lower_index: int = math.floor(position)
    upper_index: int = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction: float = position - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def parse_theta_values(rows: list[dict[str, str]]) -> list[float]:
    """Return parsed theta values from safellm4se.sampling rows.

    Args:
        rows: Sampling rows to parse.

    Returns:
        Parsed theta values.

    Raises:
        ValueError: If a theta value is empty, non-numeric, or not finite.
    """
    return _numeric_column(rows, "theta")


def infer_theta_type(rows: list[dict[str, str]], theta_values: list[float]) -> str:
    """Return the theta metric type for report calculations.

    Args:
        rows: Sampling rows for one task.
        theta_values: Parsed theta values.

    Returns:
        The metric type, either binary or continuous.

    Raises:
        ValueError: If metric_type is present with inconsistent or invalid values.
    """
    metric_types: set[str] = set()
    for row in rows:
        for fieldname in ("metric_type", "theta_type"):
            raw_metric_type: str = row.get(fieldname, "").strip()
            if raw_metric_type:
                metric_types.add(raw_metric_type.casefold())
    if len(metric_types) > 1:
        formatted_values: str = ", ".join(sorted(metric_types))
        raise ValueError(f"Inconsistent values for 'metric_type': {formatted_values}.")
    if metric_types:
        metric_type: str = next(iter(metric_types))
        if metric_type not in {"binary", "continuous"}:
            raise ValueError("metric_type must be either binary or continuous.")
        return metric_type
    if all(value in {0.0, 1.0} for value in theta_values):
        return "binary"
    return "continuous"


def _numeric_column(rows: list[dict[str, str]], fieldname: str) -> list[float]:
    """Return parsed float values from a required numeric CSV column.

    Args:
        rows: Input rows to parse.
        fieldname: Numeric field name to read.

    Returns:
        Parsed float values.

    Raises:
        ValueError: If a field value is empty, non-numeric, or not finite.
    """
    values: list[float] = []
    for row_index, row in enumerate(rows, start=1):
        raw_value: str = row.get(fieldname, "")
        try:
            value: float = float(raw_value)
        except ValueError as exception:
            raise ValueError(
                f"Invalid numeric value for '{fieldname}' in filtered row "
                f"{row_index}: {raw_value!r}."
            ) from exception
        if not math.isfinite(value):
            raise ValueError(
                f"Invalid non-finite value for '{fieldname}' in filtered row "
                f"{row_index}: {raw_value!r}."
            )
        values.append(value)
    return values


def integer_column_sum(rows: list[dict[str, str]], fieldname: str) -> int:
    """Return the sum of parsed integer values from a required CSV column.

    Args:
        rows: Input rows to parse.
        fieldname: Integer field name to read.

    Returns:
        Sum of parsed integer values.

    Raises:
        ValueError: If a field value is empty or cannot be parsed as an integer.
    """
    total: int = 0
    for row_index, row in enumerate(rows, start=1):
        raw_value: str = row.get(fieldname, "")
        try:
            total += int(raw_value)
        except ValueError as exception:
            raise ValueError(
                f"Invalid integer value for '{fieldname}' in filtered row "
                f"{row_index}: {raw_value!r}."
            ) from exception
    return total


def coefficient_of_variation(
    mean_value: float,
    standard_deviation: float,
) -> float:
    """Return the coefficient of variation expressed as a percentage.

    Args:
        mean_value: Arithmetic mean of the sample.
        standard_deviation: Sample standard deviation.

    Returns:
        The coefficient of variation in percent, or NaN when the mean is zero.
    """
    if mean_value == 0.0:
        return math.nan
    return standard_deviation / abs(mean_value) * 100.0
