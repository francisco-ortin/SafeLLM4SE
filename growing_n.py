"""Print incremental SafeLLM4SE reports for task-id-53.

This script reads output/measurements.csv, builds reports for the first N
measurements of task-id-53 where N grows from 1 to 30, and writes the report
rows to standard output in CSV format.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT: Path = Path(__file__).resolve().parent
INPUT_PATH: Path = PROJECT_ROOT / "output" / "measurements.csv"
TASK_ID: str = "task-id-67"
CONFIDENCE_LEVEL: float = 0.95
CI_METHOD: str = "auto"


def configure_import_path() -> None:
    """Add the local src directory to the Python import path.

    This function allows the script to run directly from the project root.
    """
    source_path: Path = PROJECT_ROOT / "src"
    sys.path.insert(0, str(source_path))


def build_incremental_report_rows() -> list[dict[str, Any]]:
    """Build report rows for incremental sample sizes.

    Returns:
        Report rows for N values from 1 to 30.

    Raises:
        FileNotFoundError: If output/measurements.csv does not exist.
        ValueError: If the input CSV is invalid or has fewer than 30 matching rows.
    """
    from safellm4se.reporting.metrics import build_report_row
    from safellm4se.reporting.reader import read_task_rows

    task_rows: list[dict[str, str]] = read_task_rows(INPUT_PATH, TASK_ID)
    report_rows: list[dict[str, Any]] = []
    for sample_size in range(1, len(task_rows) + 1):
        report_rows.append(
            build_report_row(
                task_rows[:sample_size],
                CONFIDENCE_LEVEL,
                CI_METHOD,
            )
        )
    return report_rows


def print_report_rows(report_rows: list[dict[str, Any]]) -> None:
    """Print report rows to standard output in CSV format.

    Args:
        report_rows: Report rows to print.

    Raises:
        OSError: If writing to standard output fails.
    """
    from safellm4se.reporting.metrics import REPORT_FIELDS

    writer: csv.DictWriter[str] = csv.DictWriter(
        sys.stdout,
        fieldnames=REPORT_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(report_rows)


def main() -> None:
    """Run the incremental reporting workflow.

    Raises:
        FileNotFoundError: If output/measurements.csv does not exist.
        OSError: If writing to standard output fails.
        ValueError: If report generation fails for the selected input rows.
    """
    configure_import_path()
    report_rows: list[dict[str, Any]] = build_incremental_report_rows()
    print_report_rows(report_rows)


if __name__ == "__main__":
    main()
