"""High-level report generation workflow for SafeLLM4SE sampling outputs."""

from typing import Any

from loguru import logger

from safellm4se.reporting.metrics import build_report_row, parse_theta_values
from safellm4se.reporting.models import ReportingSettings
from safellm4se.reporting.plots import generate_requested_plots
from safellm4se.reporting.reader import read_task_rows
from safellm4se.reporting.writer import write_report


def generate_report(settings: ReportingSettings) -> dict[str, Any]:
    """Generate a sampling report CSV.

    Args:
        settings: Runtime settings for report generation.

    Returns:
        The generated report row.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        OSError: If the output CSV file cannot be written.
        ValueError: If the input data is invalid or inconsistent.
    """
    logger.info("Reading sampling measurements from {}.", settings.input_path)
    rows: list[dict[str, str]] = read_task_rows(
        settings.input_path,
        settings.task_id,
    )
    theta_values: list[float] = parse_theta_values(rows)
    logger.info("Generating report from {} filtered rows.", len(rows))
    report_row: dict[str, Any] = build_report_row(
        rows,
        settings.confidence_level,
        settings.ci_method,
    )
    write_report(settings.output_path, report_row)
    logger.info("Report written to {}.", settings.output_path)
    generated_plot_paths = generate_requested_plots(theta_values, settings)
    for generated_plot_path in generated_plot_paths:
        logger.info("Plot written to {}.", generated_plot_path)
    return report_row
