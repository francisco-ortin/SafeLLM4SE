"""Data structures used by the reporting workflow."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportingSettings:
    """Runtime settings for a sampling report generation run."""

    input_path: Path  # CSV file containing sampling measurements.
    output_path: Path  # CSV file where the generated report is written.
    boxplot_path: Path | None  # Optional SVG file where the boxplot is written.
    violin_path: Path | None  # Optional SVG file where the violin plot is written.
    ecdf_path: Path | None  # Optional SVG file where the ECDF plot is written.
    # Optional SVG file where the raincloud plot is written.
    raincloud_path: Path | None
    kde_path: Path | None  # Optional SVG file where the KDE plot is written.
    task_id: str  # Task identifier used to filter input measurements.
    task_name: str | None  # Optional task display name used in plots.
    confidence_level: float  # Confidence level used for confidence intervals.
    ci_method: str  # Requested confidence interval method for continuous metrics.
