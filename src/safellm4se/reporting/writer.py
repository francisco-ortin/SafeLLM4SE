"""CSV output writing for SafeLLM4SE sampling reports."""

from pathlib import Path
from typing import Any

from safellm4se.csv_output import append_csv_row
from safellm4se.reporting.metrics import REPORT_FIELDS


def write_report(output_path: Path, report_row: dict[str, Any]) -> None:
    """Append a single-row report to a CSV file.

    Args:
        output_path: Destination report CSV path.
        report_row: Report row to write.

    Raises:
        OSError: If the destination file cannot be written.
        csv.Error: If the existing destination file cannot be parsed as CSV.
    """
    append_csv_row(output_path, report_row, REPORT_FIELDS)
