"""CSV output writing for SafeLLM4SE comparison reports."""

import csv
from pathlib import Path
from typing import Any

from safellm4se.comparing.metrics import COMPARISON_FIELDS


def write_comparison_report(output_path: Path, report_row: dict[str, Any]) -> None:
    """Write a single-row comparison report CSV file.

    Args:
        output_path: Destination comparison report CSV path.
        report_row: Comparison report row to write.

    Raises:
        OSError: If the destination file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer: csv.DictWriter[str] = csv.DictWriter(
            csv_file,
            fieldnames=COMPARISON_FIELDS,
        )
        writer.writeheader()
        output_row: dict[str, Any] = {
            fieldname: report_row[fieldname] for fieldname in COMPARISON_FIELDS
        }
        writer.writerow(output_row)
