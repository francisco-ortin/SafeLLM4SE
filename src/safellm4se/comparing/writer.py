"""CSV output writing for SafeLLM4SE comparison reports."""

from pathlib import Path
from typing import Any

from safellm4se.csv_output import append_csv_row
from safellm4se.comparing.metrics import COMPARISON_FIELDS


def write_comparison_report(output_path: Path, report_row: dict[str, Any]) -> None:
    """Append a single-row comparison report to a CSV file.

    Args:
        output_path: Destination comparison report CSV path.
        report_row: Comparison report row to write.

    Raises:
        OSError: If the destination file cannot be written.
        csv.Error: If the existing destination file cannot be parsed as CSV.
    """
    append_csv_row(output_path, report_row, COMPARISON_FIELDS)
