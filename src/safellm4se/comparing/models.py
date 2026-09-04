"""Data structures used by the comparison workflow."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonSettings:
    """Runtime settings for a two-sample comparison report."""

    input_path: Path  # CSV file containing sampling measurements.
    output_path: Path  # CSV file where the generated comparison report is written.
    boxplot_path: Path | None  # Optional SVG file where the boxplot is written.
    violin_path: Path | None  # Optional SVG file where the violin plot is written.
    ecdf_path: Path | None  # Optional SVG file where the ECDF plot is written.
    # Optional SVG file where the raincloud plot is written.
    raincloud_path: Path | None
    kde_path: Path | None  # Optional SVG file where the KDE plot is written.
    task_id_1: str  # First task identifier used to filter measurements.
    task_id_2: str  # Second task identifier used to filter measurements.
    task_name_1: str | None  # Optional first task display name used in plots.
    task_name_2: str | None  # Optional second task display name used in plots.
    test_type: str  # Experimental design used for the statistical comparison.
    confidence_level: float  # Confidence level used for difference intervals.


@dataclass(frozen=True)
class SampleSummary:
    """Descriptive statistics and metadata for one filtered sample."""

    task_id: str  # Task identifier represented by the sample.
    model_name: str  # Human-readable model family or provider label.
    model_id: str  # Provider-specific model identifier used for the call.
    temperature: str  # Optional sampling temperature used by all observations.
    sample_size: int  # Number of observations in the sample.
    prompt_tokens: int  # Sum of prompt tokens across observations.
    completion_tokens: int  # Sum of completion tokens across observations.
    total_tokens: int  # Sum of prompt and completion tokens.
    theta_values: list[float]  # Parsed theta observations.
    theta_type: str  # Metric type represented by theta values.
    theta_mean: float  # Arithmetic mean of theta values.
    theta_median: float  # Median of theta values.
    theta_minimum: float  # Minimum theta value.
    theta_maximum: float  # Maximum theta value.
    standard_deviation: float  # Sample standard deviation of theta values.
    coefficient_of_variation: float  # Coefficient of variation in percent.
    interquartile_range: float  # Difference between the third and first quartile.
    first_quartile: float  # First quartile of theta values.
    third_quartile: float  # Third quartile of theta values.


@dataclass(frozen=True)
class ComparisonStatistics:
    """Inferential statistics for a two-sample comparison."""

    estimated_difference: float  # Difference theta_mean_1 minus theta_mean_2.
    ci_low: float  # Lower bound of the bootstrap confidence interval.
    ci_high: float  # Upper bound of the bootstrap confidence interval.
    ci_width: float  # Width of the bootstrap confidence interval.
    confidence_level: float  # Confidence level used for the interval.
    ci_method: str  # Bootstrap interval method used by the protocol.
    statistical_test: str  # Name of the selected significance test.
    test_statistic: float  # Test statistic returned by the significance test.
    p_value: float  # P-value returned by the significance test.
    effect_size_name: str  # Name of the selected effect size.
    effect_size: float  # Numeric effect size value.
    effect_size_magnitude: str  # Qualitative effect size interpretation.
