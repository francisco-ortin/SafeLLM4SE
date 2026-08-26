"""Command-line entry point for aggregating SAFE measurement CSV files."""

from csvutils import read_measurements_csv, write_summary_csv


def compute_results(
    measurements_path: str,
    results_path: str,
) -> None:
    """Generates the aggregated results CSV from the measurements CSV."""
    measurements: list[dict[str, object]] = read_measurements_csv(measurements_path)
    write_summary_csv(measurements, results_path)


if __name__ == "__main__":
    from csvutils import get_results_file_paths

    measurements_file: str
    results_file: str
    measurements_file, results_file = get_results_file_paths()
    compute_results(measurements_file, results_file)
    print(f"Results written to {results_file}.")
