"""Confidence intervals and descriptive statistics for SAFE-style sampling."""

import math
import random
import statistics


def normal_quantile(confidence_level: float) -> float:
    """Return the two-sided normal critical value for a confidence level.
    Args:
        confidence_level: Confidence level in the open interval between 0 and 1.
    Returns:
        The normal critical value, using a fixed fallback if NormalDist fails.
    """
    alpha = 1.0 - confidence_level
    try:
        from statistics import NormalDist

        return NormalDist().inv_cdf(1.0 - alpha / 2.0)
    except Exception:
        return 1.959963984540054


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
    """
    if not values:
        return math.nan, math.nan, method
    if len(values) == 1:
        return values[0], values[0], method

    selected = "t" if method == "auto" else method
    if selected == "bootstrap":
        means = []
        for _ in range(bootstrap_samples):
            means.append(statistics.fmean(random.choice(values) for _ in values))
        means.sort()
        alpha = 1.0 - confidence_level
        low_index = max(0, int((alpha / 2.0) * len(means)))
        high_index = min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)))
        return means[low_index], means[high_index], selected

    mean_value = statistics.fmean(values)
    std_value = statistics.stdev(values)
    try:
        from scipy.stats import t

        critical = float(
            t.ppf(1.0 - (1.0 - confidence_level) / 2.0, len(values) - 1)
        )
    except Exception:
        critical = normal_quantile(confidence_level)
    margin = critical * std_value / math.sqrt(len(values))
    return mean_value - margin, mean_value + margin, selected


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
    """
    if n == 0:
        return math.nan, math.nan
    z = normal_quantile(confidence_level)
    phat = successes / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
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
    """
    if metric_type == "binary":
        successes = sum(1 for value in values if int(round(value)) == 1)
        low, high = wilson_interval(successes, len(values), confidence_level)
        return low, high, "wilson"
    return mean_confidence_interval(values, confidence_level, method, bootstrap_samples)
