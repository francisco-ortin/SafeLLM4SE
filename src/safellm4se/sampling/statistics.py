"""Compatibility exports for SAFE-style sampling statistics."""

from safellm4se.statistical_utils import (
    MAXIMUM_SHAPIRO_P_VALUE_SAMPLE_SIZE,
    MINIMUM_NORMALITY_SAMPLE_SIZE,
    NORMALITY_SIGNIFICANCE_LEVEL,
    bootstrap_mean_confidence_interval,
    can_reasonably_assume_normality,
    confidence_interval,
    mean_confidence_interval,
    normal_quantile,
    select_continuous_confidence_interval_method,
    t_critical_value,
    t_mean_confidence_interval,
    validate_bootstrap_samples,
    validate_confidence_level,
    wilson_interval,
)

__all__: list[str] = [
    "MAXIMUM_SHAPIRO_P_VALUE_SAMPLE_SIZE",
    "MINIMUM_NORMALITY_SAMPLE_SIZE",
    "NORMALITY_SIGNIFICANCE_LEVEL",
    "bootstrap_mean_confidence_interval",
    "can_reasonably_assume_normality",
    "confidence_interval",
    "mean_confidence_interval",
    "normal_quantile",
    "select_continuous_confidence_interval_method",
    "t_critical_value",
    "t_mean_confidence_interval",
    "validate_bootstrap_samples",
    "validate_confidence_level",
    "wilson_interval",
]
