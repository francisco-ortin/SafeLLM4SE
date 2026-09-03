"""High-level comparison workflow for SafeLLM4SE sampling outputs."""

from typing import Any

from loguru import logger

from safellm4se.compare.metrics import build_comparison_report_row
from safellm4se.compare.models import ComparisonSettings
from safellm4se.compare.plots import generate_requested_comparison_plots
from safellm4se.compare.writer import write_comparison_report
from safellm4se.reporting.metrics import parse_theta_values
from safellm4se.reporting.reader import read_task_rows


def generate_comparison_report(settings: ComparisonSettings) -> dict[str, Any]:
    """Generate a two-sample comparison report CSV.

    Args:
        settings: Runtime settings for comparison report generation.

    Returns:
        The generated comparison report row.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        OSError: If the output CSV file or plot files cannot be written.
        ValueError: If the input data is invalid or inconsistent.
    """
    logger.info("Reading sampling measurements from {}.", settings.input_path)
    rows_1: list[dict[str, str]] = read_task_rows(
        settings.input_path,
        settings.task_id_1,
    )
    rows_2: list[dict[str, str]] = read_task_rows(
        settings.input_path,
        settings.task_id_2,
    )
    logger.info(
        "Generating comparison report from {} and {} filtered rows.",
        len(rows_1),
        len(rows_2),
    )
    report_row: dict[str, Any] = build_comparison_report_row(
        rows_1,
        rows_2,
        settings.test_type,
        settings.confidence_level,
    )
    write_comparison_report(settings.output_path, report_row)
    logger.info("Comparison report written to {}.", settings.output_path)
    generated_plot_paths = generate_requested_comparison_plots(
        parse_theta_values(rows_1),
        parse_theta_values(rows_2),
        settings,
    )
    for generated_plot_path in generated_plot_paths:
        logger.info("Comparison plot written to {}.", generated_plot_path)
    return report_row
