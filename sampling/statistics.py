"""Confidence intervals and normality checks for SAFE-style sampling."""

import math
import random
import statistics
from typing import Any

MINIMUM_NORMALITY_SAMPLE_SIZE: int = 3
MAXIMUM_SHAPIRO_P_VALUE_SAMPLE_SIZE: int = 5_000
NORMALITY_SIGNIFICANCE_LEVEL: float = 0.05


def normal_quantile(confidence_level: float) -> float:
    """Return the two-sided normal critical value for a confidence level.
    Args:
        confidence_level: Confidence level in the open interval between 0 and 1.
    Returns:
        The normal critical value, using a fixed fallback if NormalDist fails.
    Raises:
        ValueError: If the confidence level is outside the open interval (0, 1).
    """
    validate_confidence_level(confidence_level)
    alpha = 1.0 - confidence_level
    try:
        from statistics import NormalDist

        return NormalDist().inv_cdf(1.0 - alpha / 2.0)
    except Exception:
        return 1.959963984540054


def validate_confidence_level(confidence_level: float) -> None:
    """Validate that a confidence level can be used for interval estimation.
    Args:
        confidence_level: Confidence level to validate.
    Returns:
        None.
    Raises:
        ValueError: If the confidence level is outside the open interval (0, 1).
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")


def validate_bootstrap_samples(bootstrap_samples: int) -> None:
    """Validate the number of bootstrap resamples.
    Args:
        bootstrap_samples: Number of bootstrap resamples to validate.
    Returns:
        None.
    Raises:
        ValueError: If the number of bootstrap resamples is less than one.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be greater than 0.")


def can_reasonably_assume_normality(values: list[float]) -> bool:
    """Return whether values pass the framework normality check.
    Args:
        values: Observed continuous metric values.
    Returns:
        True when the values can be treated as approximately normal; otherwise,
        False.
    """
    finite_values: list[float] = _finite_values(values)
    if len(finite_values) < MINIMUM_NORMALITY_SAMPLE_SIZE:
        return False
    if _has_no_variability(finite_values):
        return True
    if len(finite_values) <= MAXIMUM_SHAPIRO_P_VALUE_SAMPLE_SIZE:
        return _passes_shapiro_wilk_test(finite_values)
    return _passes_anderson_darling_normality_test(finite_values)


def mean_confidence_interval(
    values: list[float],
    confidence_level: float,
    method: str,
    bootstrap_samples: int,
) -> tuple[float, float, str]:
    """Return a confidence interval for the arithmetic mean.
    Args:
        values: Observed continuous metric values.
        confidence_level: Confidence level used for interval estimation.
        method: Requested interval method, such as auto, t, or bootstrap.
        bootstrap_samples: Number of bootstrap resamples to draw when requested.
    Returns:
        A tuple containing the lower bound, upper bound, and selected method.
    Raises:
        ValueError: If the confidence level, method, or bootstrap sample count is
            invalid.
    """
    validate_confidence_level(confidence_level)
    _validate_confidence_interval_method(method)
    if not values:
        return math.nan, math.nan, method
    if len(values) == 1:
        return values[0], values[0], method

    selected: str = select_continuous_confidence_interval_method(values, method)
    if selected == "bootstrap":
        low, high = bootstrap_mean_confidence_interval(
            values,
            confidence_level,
            bootstrap_samples,
        )
        return low, high, selected

    low, high = t_mean_confidence_interval(values, confidence_level)
    return low, high, selected


def select_continuous_confidence_interval_method(
    values: list[float],
    method: str,
) -> str:
    """Return the confidence interval method for continuous values.
    Args:
        values: Observed continuous metric values.
        method: User-requested method, using auto to enable normality checking.
    Returns:
        The selected concrete method, either t or bootstrap.
    Raises:
        ValueError: If the requested method is unsupported.
    """
    _validate_confidence_interval_method(method)
    if method != "auto":
        return method
    if can_reasonably_assume_normality(values):
        return "t"
    return "bootstrap"


def t_mean_confidence_interval(
    values: list[float],
    confidence_level: float,
) -> tuple[float, float]:
    """Return a t-based confidence interval for the arithmetic mean.
    Args:
        values: Observed continuous metric values.
        confidence_level: Confidence level used for interval estimation.
    Returns:
        A tuple containing the lower and upper confidence bounds.
    Raises:
        ValueError: If the confidence level is invalid.
        statistics.StatisticsError: If fewer than two values are provided.
    """
    validate_confidence_level(confidence_level)
    mean_value: float = statistics.fmean(values)
    standard_deviation: float = statistics.stdev(values)
    critical: float = t_critical_value(confidence_level, len(values) - 1)
    margin: float = critical * standard_deviation / math.sqrt(len(values))
    return mean_value - margin, mean_value + margin


def bootstrap_mean_confidence_interval(
    values: list[float],
    confidence_level: float,
    bootstrap_samples: int,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean.
    Args:
        values: Observed continuous metric values.
        confidence_level: Confidence level used for interval estimation.
        bootstrap_samples: Number of bootstrap resamples to draw.
    Returns:
        A tuple containing the lower and upper confidence bounds.
    Raises:
        ValueError: If the confidence level or bootstrap sample count is invalid.
    """
    validate_confidence_level(confidence_level)
    validate_bootstrap_samples(bootstrap_samples)
    bootstrap_means: list[float] = [
        statistics.fmean(random.choice(values) for _ in values)
        for _ in range(bootstrap_samples)
    ]
    bootstrap_means.sort()
    alpha: float = 1.0 - confidence_level
    low_index: int = _quantile_index(alpha / 2.0, len(bootstrap_means))
    high_index: int = _quantile_index(1.0 - alpha / 2.0, len(bootstrap_means))
    return bootstrap_means[low_index], bootstrap_means[high_index]


def wilson_interval(
    successes: int,
    n: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Return a Wilson confidence interval for a binary proportion.
    Args:
        successes: Number of successful binary outcomes.
        n: Total number of binary observations.
        confidence_level: Confidence level used for interval estimation.
    Returns:
        A tuple containing the lower and upper confidence bounds.
    Raises:
        ValueError: If counts or the confidence level are invalid.
    """
    validate_confidence_level(confidence_level)
    if successes < 0:
        raise ValueError("successes cannot be negative.")
    if n < 0:
        raise ValueError("n cannot be negative.")
    if successes > n:
        raise ValueError("successes cannot be greater than n.")
    if n == 0:
        return math.nan, math.nan
    z_score: float = normal_quantile(confidence_level)
    proportion: float = successes / n
    denominator: float = 1.0 + z_score * z_score / n
    center: float = (
        proportion + z_score * z_score / (2.0 * n)
    ) / denominator
    margin: float = (
        z_score
        * math.sqrt(
            (
                proportion * (1.0 - proportion)
                + z_score * z_score / (4.0 * n)
            )
            / n
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def confidence_interval(
    values: list[float],
    metric_type: str,
    confidence_level: float,
    method: str,
    bootstrap_samples: int,
) -> tuple[float, float, str]:
    """Return a confidence interval for binary or continuous metrics.
    Args:
        values: Observed metric values.
        metric_type: Metric type, with binary selecting a Wilson interval.
        confidence_level: Confidence level used for interval estimation.
        method: Requested interval method for continuous values.
        bootstrap_samples: Number of bootstrap resamples for bootstrap intervals.
    Returns:
        A tuple containing the lower bound, upper bound, and selected method.
    Raises:
        ValueError: If the metric type, confidence level, method, or bootstrap
            sample count is invalid.
    """
    if metric_type == "binary":
        successes: int = sum(1 for value in values if int(round(value)) == 1)
        low, high = wilson_interval(successes, len(values), confidence_level)
        return low, high, "wilson"
    if metric_type != "continuous":
        raise ValueError("metric_type must be either binary or continuous.")
    return mean_confidence_interval(
        values,
        confidence_level,
        method,
        bootstrap_samples,
    )


def t_critical_value(confidence_level: float, degrees_of_freedom: int) -> float:
    """Return a two-sided Student t critical value.
    Args:
        confidence_level: Confidence level used for interval estimation.
        degrees_of_freedom: Degrees of freedom for the t distribution.
    Returns:
        The t critical value, or a normal approximation if SciPy is unavailable.
    Raises:
        ValueError: If the confidence level or degrees of freedom is invalid.
    """
    validate_confidence_level(confidence_level)
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be greater than 0.")
    try:
        from scipy.stats import t

        return float(
            t.ppf(1.0 - (1.0 - confidence_level) / 2.0, degrees_of_freedom)
        )
    except Exception:
        return normal_quantile(confidence_level)


def _passes_shapiro_wilk_test(values: list[float]) -> bool:
    """Return whether values pass the Shapiro-Wilk normality test.
    Args:
        values: Finite continuous metric values with at least three elements.
    Returns:
        True when the Shapiro-Wilk p-value does not reject normality; otherwise,
        False.
    """
    try:
        from scipy.stats import shapiro

        _, p_value = shapiro(values)
    except Exception:
        return False
    return bool(
        math.isfinite(float(p_value))
        and p_value >= NORMALITY_SIGNIFICANCE_LEVEL
    )


def _passes_anderson_darling_normality_test(values: list[float]) -> bool:
    """Return whether values pass the Anderson-Darling normality test.
    Args:
        values: Finite continuous metric values with more than 5000 elements.
    Returns:
        True when the test statistic does not exceed the 5 percent critical
        value; otherwise, False.
    """
    try:
        from scipy.stats import anderson

        result: Any = anderson(values, dist="norm")
    except Exception:
        return False
    critical_value: float | None = _anderson_darling_critical_value(
        list(result.critical_values),
        list(result.significance_level),
        NORMALITY_SIGNIFICANCE_LEVEL,
    )
    if critical_value is None:
        return False
    return bool(float(result.statistic) <= critical_value)


def _anderson_darling_critical_value(
    critical_values: list[float],
    significance_levels: list[float],
    target_alpha: float,
) -> float | None:
    """Return the Anderson-Darling critical value closest to target alpha.
    Args:
        critical_values: Critical values returned by SciPy.
        significance_levels: Percent significance levels returned by SciPy.
        target_alpha: Desired significance level expressed as a fraction.
    Returns:
        The closest critical value, or None when no critical values are available.
    """
    if not critical_values or not significance_levels:
        return None
    target_percent: float = target_alpha * 100.0
    indexed_levels: list[tuple[int, float]] = list(enumerate(significance_levels))
    closest_index, _ = min(
        indexed_levels,
        key=lambda indexed_level: abs(float(indexed_level[1]) - target_percent),
    )
    return float(critical_values[closest_index])


def _finite_values(values: list[float]) -> list[float]:
    """Return only finite values from a metric sample.
    Args:
        values: Observed metric values.
    Returns:
        A list containing only finite floating-point values.
    """
    return [float(value) for value in values if math.isfinite(float(value))]


def _has_no_variability(values: list[float]) -> bool:
    """Return whether all values are numerically identical.
    Args:
        values: Observed finite metric values.
    Returns:
        True when all values match the first value; otherwise, False.
    """
    first_value: float = values[0]
    return all(value == first_value for value in values)


def _quantile_index(probability: float, sample_size: int) -> int:
    """Return a bounded index for a sorted empirical quantile.
    Args:
        probability: Desired quantile probability.
        sample_size: Number of sorted observations.
    Returns:
        A valid zero-based index into the sorted observations.
    Raises:
        ValueError: If sample_size is less than one.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be greater than 0.")
    return min(sample_size - 1, max(0, int(probability * sample_size)))


def _validate_confidence_interval_method(method: str) -> None:
    """Validate a requested confidence interval method.
    Args:
        method: Requested confidence interval method.
    Returns:
        None.
    Raises:
        ValueError: If the method is unsupported.
    """
    if method not in {"auto", "t", "bootstrap"}:
        raise ValueError("method must be auto, t, or bootstrap.")
