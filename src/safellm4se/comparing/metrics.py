"""Comparison metric calculations for SafeLLM4SE sampling measurements."""

import math
import random
import statistics
from datetime import datetime
from typing import Any

from safellm4se.sampling.config import config
from safellm4se.statistical_utils import validate_bootstrap_samples, validate_confidence_level

from safellm4se.comparing.models import ComparisonStatistics, SampleSummary
from safellm4se.reporting.metrics import (
    coefficient_of_variation,
    infer_theta_type,
    integer_column_sum,
    parse_theta_values,
    percentile,
)

COMPARISON_FIELDS: list[str] = [
    "date",
    "time",
    "task_id_1",
    "task_id_2",
    "model_name_1",
    "model_name_2",
    "model_id_1",
    "model_id_2",
    "temperature_1",
    "temperature_2",
    "N_1",
    "N_2",
    "prompt_tokens_1",
    "prompt_tokens_2",
    "completion_tokens_1",
    "completion_tokens_2",
    "total_tokens_1",
    "total_tokens_2",
    "theta_mean_1",
    "theta_mean_2",
    "theta_median_1",
    "theta_median_2",
    "theta_min_1",
    "theta_min_2",
    "theta_max_1",
    "theta_max_2",
    "theta_type",
    "sd_1",
    "sd_2",
    "cv_1",
    "cv_2",
    "iqr_1",
    "iqr_2",
    "q1_1",
    "q1_2",
    "q3_1",
    "q3_2",
    "test_type",
    "estimated_difference",
    "ci_method",
    "ci_confidence-level",
    "ci_low",
    "ci_high",
    "ci_width",
    "statistical_test",
    "test_statistic",
    "p_value",
    "effect_size_name",
    "effect_size",
    "effect_size_magnitude",
]

EFFECT_SIZE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.474, "large"),
    (0.33, "medium"),
    (0.147, "small"),
)


def build_comparison_report_row(
    rows_1: list[dict[str, str]],
    rows_2: list[dict[str, str]],
    test_type: str,
    confidence_level: float,
) -> dict[str, Any]:
    """Build one comparison report row from two filtered samples.

    Args:
        rows_1: Sampling rows for the first task.
        rows_2: Sampling rows for the second task.
        test_type: Experimental design, either paired or independent.
        confidence_level: Confidence level used for bootstrap intervals.

    Returns:
        A CSV-ready comparison row with descriptive and inferential statistics.

    Raises:
        ValueError: If the design, confidence level, or input data are invalid.
    """
    summary_1: SampleSummary = summarize_sample(rows_1)
    summary_2: SampleSummary = summarize_sample(rows_2)
    _validate_common_theta_type(summary_1, summary_2)
    comparison_statistics: ComparisonStatistics = compare_samples(
        rows_1,
        rows_2,
        summary_1,
        summary_2,
        test_type,
        confidence_level,
    )
    timestamp: datetime = datetime.now()
    row: dict[str, Any] = {
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "theta_type": summary_1.theta_type,
        "test_type": test_type,
    }
    row.update(_summary_columns(summary_1, "1"))
    row.update(_summary_columns(summary_2, "2"))
    row.update(_comparison_columns(comparison_statistics))
    return {fieldname: row[fieldname] for fieldname in COMPARISON_FIELDS}


def summarize_sample(rows: list[dict[str, str]]) -> SampleSummary:
    """Return descriptive statistics and metadata for one filtered sample.

    Args:
        rows: Sampling rows for one task.

    Returns:
        Summary statistics and metadata for the sample.

    Raises:
        ValueError: If required numeric values cannot be parsed.
    """
    theta_values: list[float] = parse_theta_values(rows)
    theta_type: str = infer_theta_type(rows, theta_values)
    first_quartile: float = percentile(theta_values, 0.25)
    third_quartile: float = percentile(theta_values, 0.75)
    mean_value: float = statistics.fmean(theta_values)
    standard_deviation: float = (
        statistics.stdev(theta_values) if len(theta_values) > 1 else 0.0
    )
    prompt_tokens: int = integer_column_sum(rows, "prompt_tokens")
    completion_tokens: int = integer_column_sum(rows, "completion_tokens")
    return SampleSummary(
        task_id=rows[0]["task_id"],
        model_name=rows[0]["model_name"],
        model_id=rows[0]["model_id"],
        temperature=rows[0].get("temperature", ""),
        sample_size=len(rows),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        theta_values=theta_values,
        theta_type=theta_type,
        theta_mean=mean_value,
        theta_median=statistics.median(theta_values),
        theta_minimum=min(theta_values),
        theta_maximum=max(theta_values),
        standard_deviation=standard_deviation,
        coefficient_of_variation=coefficient_of_variation(
            mean_value,
            standard_deviation,
        ),
        interquartile_range=third_quartile - first_quartile,
        first_quartile=first_quartile,
        third_quartile=third_quartile,
    )


def compare_samples(
    rows_1: list[dict[str, str]],
    rows_2: list[dict[str, str]],
    summary_1: SampleSummary,
    summary_2: SampleSummary,
    test_type: str,
    confidence_level: float,
) -> ComparisonStatistics:
    """Return inferential statistics for a two-sample comparison.

    Args:
        rows_1: Sampling rows for the first task.
        rows_2: Sampling rows for the second task.
        summary_1: Descriptive summary for the first sample.
        summary_2: Descriptive summary for the second sample.
        test_type: Experimental design, either paired or independent.
        confidence_level: Confidence level used for bootstrap intervals.

    Returns:
        Inferential statistics selected by the SafeLLM4SE protocol.

    Raises:
        ValueError: If the design or confidence level is invalid.
    """
    validate_confidence_level(confidence_level)
    validate_bootstrap_samples(config.bootstrap_samples)
    if test_type == "independent":
        return _independent_comparison(summary_1, summary_2, confidence_level)
    if test_type == "paired":
        paired_values: list[tuple[float, float]] = paired_theta_values(rows_1, rows_2)
        return _paired_comparison(
            paired_values,
            summary_1,
            summary_2,
            confidence_level,
        )
    raise ValueError("test_type must be either paired or independent.")


def paired_theta_values(
    rows_1: list[dict[str, str]],
    rows_2: list[dict[str, str]],
) -> list[tuple[float, float]]:
    """Return theta values aligned by execution number for paired tests.

    Args:
        rows_1: Sampling rows for the first task.
        rows_2: Sampling rows for the second task.

    Returns:
        Theta value pairs sorted by execution number.

    Raises:
        ValueError: If execution numbers are missing, duplicated, or unmatched.
    """
    values_1: dict[int, float] = _theta_by_execution_number(rows_1, "task_id_1")
    values_2: dict[int, float] = _theta_by_execution_number(rows_2, "task_id_2")
    if set(values_1) != set(values_2):
        only_1: list[int] = sorted(set(values_1) - set(values_2))
        only_2: list[int] = sorted(set(values_2) - set(values_1))
        raise ValueError(
            "Paired comparisons require matching execution_number values. "
            f"Only in task_id_1: {only_1}; only in task_id_2: {only_2}."
        )
    return [
        (values_1[execution_number], values_2[execution_number])
        for execution_number in sorted(values_1)
    ]


def cliffs_delta(values_1: list[float], values_2: list[float]) -> float:
    """Return Cliff's delta for two independent samples.

    Args:
        values_1: First sample values.
        values_2: Second sample values.

    Returns:
        Cliff's delta in the interval [-1, 1].

    Raises:
        ValueError: If either sample is empty.
    """
    if not values_1 or not values_2:
        raise ValueError("Both samples must contain at least one value.")
    greater_count: int = 0
    lower_count: int = 0
    for value_1 in values_1:
        for value_2 in values_2:
            if value_1 > value_2:
                greater_count += 1
            elif value_1 < value_2:
                lower_count += 1
    return (greater_count - lower_count) / (len(values_1) * len(values_2))


def rank_biserial_correlation(paired_values: list[tuple[float, float]]) -> float:
    """Return matched-pairs rank-biserial correlation.

    Args:
        paired_values: Matched theta value pairs.

    Returns:
        Rank-biserial correlation in the interval [-1, 1], or 0 when all
        non-zero signed ranks vanish.
    """
    differences: list[float] = [
        value_1 - value_2
        for value_1, value_2 in paired_values
        if value_1 != value_2
    ]
    if not differences:
        return 0.0
    ranks: list[float] = _average_ranks([abs(value) for value in differences])
    positive_rank_sum: float = sum(
        rank for rank, difference in zip(ranks, differences) if difference > 0.0
    )
    negative_rank_sum: float = sum(
        rank for rank, difference in zip(ranks, differences) if difference < 0.0
    )
    total_rank_sum: float = positive_rank_sum + negative_rank_sum
    if total_rank_sum == 0.0:
        return 0.0
    return (positive_rank_sum - negative_rank_sum) / total_rank_sum


def effect_size_magnitude(effect_size: float) -> str:
    """Return a conventional qualitative interpretation for an effect size.

    Args:
        effect_size: Non-parametric effect size in the interval [-1, 1].

    Returns:
        A magnitude label: negligible, small, medium, or large.
    """
    absolute_effect_size: float = abs(effect_size)
    for threshold, label in EFFECT_SIZE_THRESHOLDS:
        if absolute_effect_size >= threshold:
            return label
    return "negligible"


def _independent_comparison(
    summary_1: SampleSummary,
    summary_2: SampleSummary,
    confidence_level: float,
) -> ComparisonStatistics:
    """Return the SafeLLM4SE independent-sample comparison statistics.

    Args:
        summary_1: Descriptive summary for the first sample.
        summary_2: Descriptive summary for the second sample.
        confidence_level: Confidence level used for bootstrap intervals.

    Returns:
        Inferential statistics for the independent-sample design.
    """
    ci_low, ci_high = _bootstrap_independent_difference_ci(
        summary_1.theta_values,
        summary_2.theta_values,
        confidence_level,
        config.bootstrap_samples,
    )
    test_statistic, p_value = _mann_whitney_u_test(
        summary_1.theta_values,
        summary_2.theta_values,
    )
    effect_size: float = cliffs_delta(summary_1.theta_values, summary_2.theta_values)
    return ComparisonStatistics(
        estimated_difference=summary_1.theta_mean - summary_2.theta_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_width=ci_high - ci_low,
        confidence_level=confidence_level * 100.0,
        ci_method="bootstrap_difference",
        statistical_test="Mann-Whitney U",
        test_statistic=test_statistic,
        p_value=p_value,
        effect_size_name="Cliff's delta",
        effect_size=effect_size,
        effect_size_magnitude=effect_size_magnitude(effect_size),
    )


def _paired_comparison(
    paired_values: list[tuple[float, float]],
    summary_1: SampleSummary,
    summary_2: SampleSummary,
    confidence_level: float,
) -> ComparisonStatistics:
    """Return the SafeLLM4SE paired-sample comparison statistics.

    Args:
        paired_values: Matched theta value pairs.
        summary_1: Descriptive summary for the first sample.
        summary_2: Descriptive summary for the second sample.
        confidence_level: Confidence level used for bootstrap intervals.

    Returns:
        Inferential statistics for the paired-sample design.
    """
    ci_low, ci_high = _bootstrap_paired_difference_ci(
        paired_values,
        confidence_level,
        config.bootstrap_samples,
    )
    test_statistic, p_value = _wilcoxon_signed_rank_test(paired_values)
    effect_size: float = rank_biserial_correlation(paired_values)
    return ComparisonStatistics(
        estimated_difference=summary_1.theta_mean - summary_2.theta_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_width=ci_high - ci_low,
        confidence_level=confidence_level * 100.0,
        ci_method="paired_bootstrap_difference",
        statistical_test="Wilcoxon signed-rank",
        test_statistic=test_statistic,
        p_value=p_value,
        effect_size_name="Matched-pairs rank-biserial correlation",
        effect_size=effect_size,
        effect_size_magnitude=effect_size_magnitude(effect_size),
    )


def _bootstrap_independent_difference_ci(
    values_1: list[float],
    values_2: list[float],
    confidence_level: float,
    bootstrap_samples: int,
) -> tuple[float, float]:
    """Return a bootstrap CI for the independent mean difference.

    Args:
        values_1: First sample values.
        values_2: Second sample values.
        confidence_level: Confidence level used for interval estimation.
        bootstrap_samples: Number of bootstrap resamples to draw.

    Returns:
        Lower and upper percentile bootstrap bounds.
    """
    differences: list[float] = [
        statistics.fmean(random.choice(values_1) for _ in values_1)
        - statistics.fmean(random.choice(values_2) for _ in values_2)
        for _ in range(bootstrap_samples)
    ]
    return _percentile_interval(differences, confidence_level)


def _bootstrap_paired_difference_ci(
    paired_values: list[tuple[float, float]],
    confidence_level: float,
    bootstrap_samples: int,
) -> tuple[float, float]:
    """Return a bootstrap CI for the paired mean difference.

    Args:
        paired_values: Matched theta value pairs.
        confidence_level: Confidence level used for interval estimation.
        bootstrap_samples: Number of bootstrap resamples to draw.

    Returns:
        Lower and upper percentile bootstrap bounds.
    """
    differences: list[float] = []
    for _ in range(bootstrap_samples):
        resampled_pairs: list[tuple[float, float]] = [
            random.choice(paired_values) for _ in paired_values
        ]
        differences.append(
            statistics.fmean(
                value_1 - value_2 for value_1, value_2 in resampled_pairs
            )
        )
    return _percentile_interval(differences, confidence_level)


def _mann_whitney_u_test(
    values_1: list[float],
    values_2: list[float],
) -> tuple[float, float]:
    """Return the Mann-Whitney U statistic and p-value.

    Args:
        values_1: First sample values.
        values_2: Second sample values.

    Returns:
        A tuple containing the U statistic and p-value.
    """
    try:
        from scipy.stats import mannwhitneyu

        result = mannwhitneyu(values_1, values_2, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return _mann_whitney_u_normal_approximation(values_1, values_2)


def _wilcoxon_signed_rank_test(
    paired_values: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return the Wilcoxon signed-rank statistic and p-value.

    Args:
        paired_values: Matched theta value pairs.

    Returns:
        A tuple containing the signed-rank statistic and p-value.
    """
    differences: list[float] = [
        value_1 - value_2
        for value_1, value_2 in paired_values
        if value_1 != value_2
    ]
    if not differences:
        return 0.0, 1.0
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return _wilcoxon_normal_approximation(differences)


def _mann_whitney_u_normal_approximation(
    values_1: list[float],
    values_2: list[float],
) -> tuple[float, float]:
    """Return Mann-Whitney U with a normal-approximation p-value.

    Args:
        values_1: First sample values.
        values_2: Second sample values.

    Returns:
        A tuple containing the U statistic and approximated p-value.
    """
    combined_values: list[float] = values_1 + values_2
    ranks: list[float] = _average_ranks(combined_values)
    rank_sum_1: float = sum(ranks[: len(values_1)])
    sample_size_1: int = len(values_1)
    sample_size_2: int = len(values_2)
    u_statistic: float = rank_sum_1 - sample_size_1 * (sample_size_1 + 1) / 2.0
    mean_u: float = sample_size_1 * sample_size_2 / 2.0
    standard_deviation_u: float = math.sqrt(
        sample_size_1 * sample_size_2 * (sample_size_1 + sample_size_2 + 1) / 12.0
    )
    if standard_deviation_u == 0.0:
        return u_statistic, 1.0
    z_score: float = (u_statistic - mean_u) / standard_deviation_u
    return u_statistic, _two_sided_normal_p_value(z_score)


def _wilcoxon_normal_approximation(differences: list[float]) -> tuple[float, float]:
    """Return Wilcoxon statistic with a normal-approximation p-value.

    Args:
        differences: Non-zero paired differences.

    Returns:
        A tuple containing the Wilcoxon statistic and approximated p-value.
    """
    ranks: list[float] = _average_ranks([abs(value) for value in differences])
    positive_rank_sum: float = sum(
        rank for rank, difference in zip(ranks, differences) if difference > 0.0
    )
    negative_rank_sum: float = sum(
        rank for rank, difference in zip(ranks, differences) if difference < 0.0
    )
    statistic: float = min(positive_rank_sum, negative_rank_sum)
    sample_size: int = len(differences)
    mean_value: float = sample_size * (sample_size + 1) / 4.0
    variance: float = sample_size * (sample_size + 1) * (2 * sample_size + 1) / 24.0
    if variance == 0.0:
        return statistic, 1.0
    z_score: float = (positive_rank_sum - mean_value) / math.sqrt(variance)
    return statistic, _two_sided_normal_p_value(z_score)


def _average_ranks(values: list[float]) -> list[float]:
    """Return average ranks for values, preserving the input order.

    Args:
        values: Numeric values to rank.

    Returns:
        One-based average ranks with tied values sharing the average rank.
    """
    indexed_values: list[tuple[int, float]] = list(enumerate(values))
    sorted_values: list[tuple[int, float]] = sorted(
        indexed_values,
        key=lambda indexed_value: indexed_value[1],
    )
    ranks: list[float] = [0.0] * len(values)
    start_index: int = 0
    while start_index < len(sorted_values):
        end_index: int = start_index
        while (
            end_index + 1 < len(sorted_values)
            and sorted_values[end_index + 1][1] == sorted_values[start_index][1]
        ):
            end_index += 1
        average_rank: float = (start_index + 1 + end_index + 1) / 2.0
        for sorted_index in range(start_index, end_index + 1):
            original_index: int = sorted_values[sorted_index][0]
            ranks[original_index] = average_rank
        start_index = end_index + 1
    return ranks


def _theta_by_execution_number(
    rows: list[dict[str, str]],
    sample_label: str,
) -> dict[int, float]:
    """Return theta values keyed by execution number.

    Args:
        rows: Sampling rows for one task.
        sample_label: User-facing sample label for error messages.

    Returns:
        Theta values keyed by execution number.

    Raises:
        ValueError: If execution numbers are missing, invalid, or duplicated.
    """
    values: dict[int, float] = {}
    for row_index, row in enumerate(rows, start=1):
        raw_execution_number: str = row.get("execution_number", "")
        try:
            execution_number: int = int(raw_execution_number)
            theta_value: float = float(row.get("theta", ""))
        except ValueError as exception:
            raise ValueError(
                f"Invalid paired data in {sample_label} filtered row {row_index}."
            ) from exception
        if execution_number in values:
            raise ValueError(
                f"Duplicated execution_number {execution_number} in {sample_label}."
            )
        values[execution_number] = theta_value
    return values


def _summary_columns(summary: SampleSummary, suffix: str) -> dict[str, Any]:
    """Return CSV columns for one sample summary.

    Args:
        summary: Sample summary to serialize.
        suffix: Output column suffix.

    Returns:
        CSV-ready sample summary columns.
    """
    return {
        f"task_id_{suffix}": summary.task_id,
        f"model_name_{suffix}": summary.model_name,
        f"model_id_{suffix}": summary.model_id,
        f"temperature_{suffix}": summary.temperature,
        f"N_{suffix}": summary.sample_size,
        f"prompt_tokens_{suffix}": summary.prompt_tokens,
        f"completion_tokens_{suffix}": summary.completion_tokens,
        f"total_tokens_{suffix}": summary.total_tokens,
        f"theta_mean_{suffix}": summary.theta_mean,
        f"theta_median_{suffix}": summary.theta_median,
        f"theta_min_{suffix}": summary.theta_minimum,
        f"theta_max_{suffix}": summary.theta_maximum,
        f"sd_{suffix}": summary.standard_deviation,
        f"cv_{suffix}": summary.coefficient_of_variation,
        f"iqr_{suffix}": summary.interquartile_range,
        f"q1_{suffix}": summary.first_quartile,
        f"q3_{suffix}": summary.third_quartile,
    }


def _comparison_columns(statistics_result: ComparisonStatistics) -> dict[str, Any]:
    """Return CSV columns for comparison statistics.

    Args:
        statistics_result: Comparison statistics to serialize.

    Returns:
        CSV-ready comparison statistics columns.
    """
    return {
        "estimated_difference": statistics_result.estimated_difference,
        "ci_method": statistics_result.ci_method,
        "ci_confidence-level": statistics_result.confidence_level,
        "ci_low": statistics_result.ci_low,
        "ci_high": statistics_result.ci_high,
        "ci_width": statistics_result.ci_width,
        "statistical_test": statistics_result.statistical_test,
        "test_statistic": statistics_result.test_statistic,
        "p_value": statistics_result.p_value,
        "effect_size_name": statistics_result.effect_size_name,
        "effect_size": statistics_result.effect_size,
        "effect_size_magnitude": statistics_result.effect_size_magnitude,
    }


def _validate_common_theta_type(
    summary_1: SampleSummary,
    summary_2: SampleSummary,
) -> None:
    """Validate that both samples use the same theta type.

    Args:
        summary_1: Descriptive summary for the first sample.
        summary_2: Descriptive summary for the second sample.

    Raises:
        ValueError: If the samples have different theta types.
    """
    if summary_1.theta_type != summary_2.theta_type:
        raise ValueError(
            "Both task samples must have the same theta_type. "
            f"task_id_1 has {summary_1.theta_type!r} and task_id_2 has "
            f"{summary_2.theta_type!r}."
        )


def _percentile_interval(
    values: list[float],
    confidence_level: float,
) -> tuple[float, float]:
    """Return percentile confidence interval bounds.

    Args:
        values: Bootstrap statistics.
        confidence_level: Confidence level used for interval estimation.

    Returns:
        Lower and upper percentile interval bounds.
    """
    alpha: float = 1.0 - confidence_level
    return percentile(values, alpha / 2.0), percentile(values, 1.0 - alpha / 2.0)


def _two_sided_normal_p_value(z_score: float) -> float:
    """Return a two-sided p-value from a standard normal z score.

    Args:
        z_score: Standard normal test statistic.

    Returns:
        Two-sided normal-approximation p-value.
    """
    return math.erfc(abs(z_score) / math.sqrt(2.0))
