"""Shared CSV output helpers for append-style report persistence."""

import csv
from pathlib import Path
from typing import Any

NULL_CSV_VALUE: str = ""


def append_csv_row(
    output_path: Path,
    row: dict[str, Any],
    preferred_fieldnames: list[str],
) -> None:
    """Append one CSV row while preserving existing rows and columns.

    Args:
        output_path: Destination CSV path.
        row: Row to append.
        preferred_fieldnames: Field names that should appear first in the CSV.

    Raises:
        OSError: If the destination file cannot be read or written.
        csv.Error: If the destination file cannot be parsed as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_fieldnames, existing_rows = _read_existing_csv(output_path)
    fieldnames: list[str] = _merged_fieldnames(
        preferred_fieldnames,
        existing_fieldnames,
        row,
    )
    output_rows: list[dict[str, Any]] = _rows_with_null_fields(
        [*existing_rows, row],
        fieldnames,
    )
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer: csv.DictWriter[str] = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)


def _read_existing_csv(output_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read an existing CSV file if it has content.

    Args:
        output_path: CSV path to read.

    Returns:
        A tuple containing existing field names and existing rows.

    Raises:
        OSError: If the existing file cannot be read.
        csv.Error: If the existing file cannot be parsed as CSV.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return [], []
    with output_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader: csv.DictReader[str] = csv.DictReader(csv_file)
        fieldnames: list[str] = list(reader.fieldnames or [])
        rows: list[dict[str, Any]] = [
            {
                fieldname: value
                for fieldname, value in row.items()
                if fieldname is not None
            }
            for row in reader
        ]
        return fieldnames, rows


def _merged_fieldnames(
    preferred_fieldnames: list[str],
    existing_fieldnames: list[str],
    row: dict[str, Any],
) -> list[str]:
    """Return CSV field names including previous and new dynamic columns.

    Args:
        preferred_fieldnames: Field names that should appear first.
        existing_fieldnames: Field names already present in the CSV.
        row: New row whose fields must be included.

    Returns:
        Ordered CSV field names.
    """
    fieldnames: list[str] = []
    for fieldname in [*preferred_fieldnames, *existing_fieldnames]:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)
    new_fieldnames: list[str] = sorted(
        fieldname
        for fieldname in row
        if fieldname not in fieldnames
    )
    fieldnames.extend(new_fieldnames)
    return fieldnames


def _rows_with_null_fields(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> list[dict[str, Any]]:
    """Return rows with blank values for missing CSV fields.

    Args:
        rows: Rows to normalize.
        fieldnames: Field names that each output row must contain.

    Returns:
        Rows with blank values for missing fields.
    """
    return [
        {
            fieldname: row.get(fieldname, NULL_CSV_VALUE)
            for fieldname in fieldnames
        }
        for row in rows
    ]
