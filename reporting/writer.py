"""CSV output writing for SafeLLM4SE sampling reports."""

import csv
from pathlib import Path
from typing import Any

from reporting.metrics import REPORT_FIELDS


def write_report(output_path: Path, report_row: dict[str, Any]) -> None:
    """Write a single-row report CSV file.

    Args:
        output_path: Destination report CSV path.
        report_row: Report row to write.

    Raises:
        OSError: If the destination file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer: csv.DictWriter[str] = csv.DictWriter(
            csv_file,
            fieldnames=REPORT_FIELDS,
        )
        writer.writeheader()
        output_row: dict[str, Any] = {
            fieldname: report_row[fieldname] for fieldname in REPORT_FIELDS
        }
        writer.writerow(output_row)
