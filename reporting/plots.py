"""SVG plot generation for SafeLLM4SE sampling reports."""

import math
import statistics
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from reporting.metrics import percentile
from reporting.models import ReportingSettings

PlotFunction = Callable[[list[float], Path, str], None]


def generate_requested_plots(
    theta_values: list[float],
    settings: ReportingSettings,
) -> list[Path]:
    """Generate every plot requested in reporting settings.

    Args:
        theta_values: Theta observations used by the report.
        settings: Runtime settings containing optional plot paths.

    Returns:
        Paths of the generated SVG files.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If any SVG file cannot be written.
        ValueError: If theta values are empty.
    """
    if not theta_values:
        raise ValueError("theta_values cannot be empty.")

    title: str = f"Theta distribution for {settings.task_id}"
    requested_plots: list[tuple[Path | None, PlotFunction]] = [
        (settings.boxplot_path, create_boxplot),
        (settings.violin_path, create_violin_plot),
        (settings.ecdf_path, create_ecdf_plot),
        (settings.raincloud_path, create_raincloud_plot),
        (settings.kde_path, create_kde_plot),
    ]
    generated_paths: list[Path] = []
    for output_path, plot_function in requested_plots:
        if output_path is None:
            continue
        logger.info("Generating plot {}.", output_path)
        plot_function(theta_values, output_path, title)
        generated_paths.append(output_path)
    return generated_paths


def create_boxplot(values: list[float], output_path: Path, title: str) -> None:
    """Create an SVG boxplot for theta observations.

    Args:
        values: Theta observations to plot.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.boxplot(
        values,
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "edgecolor": "#1f2937"},
        medianprops={"color": "#b91c1c", "linewidth": 2.0},
        whiskerprops={"color": "#1f2937"},
        capprops={"color": "#1f2937"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#f97316",
            "markeredgecolor": "#7c2d12",
            "alpha": 0.75,
        },
    )
    axis.set_title(title)
    axis.set_ylabel("theta")
    axis.set_xticks([1])
    axis.set_xticklabels(["sample"])
    _save_figure(figure, output_path)


def create_violin_plot(values: list[float], output_path: Path, title: str) -> None:
    """Create an SVG violin plot for theta observations.

    Args:
        values: Theta observations to plot.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    violin_parts: dict[str, Any] = axis.violinplot(
        values,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_edgecolor("#1f2937")
        body.set_alpha(0.8)
    axis.set_title(title)
    axis.set_ylabel("theta")
    axis.set_xticks([1])
    axis.set_xticklabels(["sample"])
    _save_figure(figure, output_path)


def create_ecdf_plot(values: list[float], output_path: Path, title: str) -> None:
    """Create an SVG empirical cumulative distribution function plot.

    Args:
        values: Theta observations to plot.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    sorted_values: list[float] = sorted(values)
    probabilities: list[float] = [
        index / len(sorted_values)
        for index in range(1, len(sorted_values) + 1)
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.step(sorted_values, probabilities, where="post", color="#2563eb")
    axis.scatter(sorted_values, probabilities, s=14, color="#f97316", zorder=3)
    axis.set_title(title)
    axis.set_xlabel("theta")
    axis.set_ylabel("ECDF")
    axis.set_ylim(0.0, 1.02)
    _save_figure(figure, output_path)


def create_raincloud_plot(values: list[float], output_path: Path, title: str) -> None:
    """Create an SVG raincloud plot for theta observations.

    Args:
        values: Theta observations to plot.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    x_values, density_values = kde_curve(values)
    maximum_density: float = max(density_values) if density_values else 1.0
    scaled_density: list[float] = [
        0.35 * density_value / maximum_density for density_value in density_values
    ]
    scatter_y_values: list[float] = [
        -0.22 + 0.08 * ((index % 7) / 6.0)
        for index, _ in enumerate(values)
    ]

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.fill_between(
        x_values,
        0.0,
        scaled_density,
        color="#9ecae1",
        alpha=0.75,
        linewidth=0.0,
    )
    axis.plot(x_values, scaled_density, color="#1d4ed8", linewidth=1.5)
    axis.boxplot(
        values,
        vert=False,
        positions=[-0.08],
        widths=[0.12],
        patch_artist=True,
        boxprops={"facecolor": "#e5e7eb", "edgecolor": "#1f2937"},
        medianprops={"color": "#b91c1c", "linewidth": 2.0},
        whiskerprops={"color": "#1f2937"},
        capprops={"color": "#1f2937"},
    )
    axis.scatter(values, scatter_y_values, s=18, color="#f97316", alpha=0.75)
    axis.set_title(title)
    axis.set_xlabel("theta")
    axis.set_yticks([])
    axis.set_ylim(-0.36, 0.42)
    _save_figure(figure, output_path)


def create_kde_plot(values: list[float], output_path: Path, title: str) -> None:
    """Create an SVG kernel density estimate plot for theta observations.

    Args:
        values: Theta observations to plot.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    x_values, density_values = kde_curve(values)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(x_values, density_values, color="#2563eb", linewidth=2.0)
    axis.fill_between(x_values, density_values, color="#9ecae1", alpha=0.45)
    axis.set_title(title)
    axis.set_xlabel("theta")
    axis.set_ylabel("density")
    _save_figure(figure, output_path)


def kde_curve(
    values: list[float],
    points: int = 200,
) -> tuple[list[float], list[float]]:
    """Return x and density values for a Gaussian KDE curve.

    Args:
        values: Theta observations used to estimate density.
        points: Number of x coordinates in the returned curve.

    Returns:
        A tuple containing x values and estimated density values.

    Raises:
        ValueError: If values is empty or points is less than two.
    """
    if not values:
        raise ValueError("values cannot be empty.")
    if points < 2:
        raise ValueError("points must be greater than 1.")
    if len(values) == 1 or min(values) == max(values):
        return _constant_density_curve(values[0], points)
    try:
        from scipy.stats import gaussian_kde

        x_values: list[float] = _plot_x_values(values, points)
        density = gaussian_kde(values)
        return x_values, [float(density(x_value)[0]) for x_value in x_values]
    except Exception:
        return _manual_gaussian_kde(values, points)


def _manual_gaussian_kde(
    values: list[float],
    points: int,
) -> tuple[list[float], list[float]]:
    """Return a Gaussian KDE curve without external statistical dependencies.

    Args:
        values: Theta observations used to estimate density.
        points: Number of x coordinates in the returned curve.

    Returns:
        A tuple containing x values and estimated density values.
    """
    x_values: list[float] = _plot_x_values(values, points)
    standard_deviation: float = statistics.stdev(values)
    bandwidth: float = 1.06 * standard_deviation * (len(values) ** (-1.0 / 5.0))
    if bandwidth <= 0.0 or not math.isfinite(bandwidth):
        return _constant_density_curve(values[0], points)
    normalizer: float = len(values) * bandwidth * math.sqrt(2.0 * math.pi)
    density_values: list[float] = [
        sum(
            math.exp(-0.5 * ((x_value - value) / bandwidth) ** 2)
            for value in values
        )
        / normalizer
        for x_value in x_values
    ]
    return x_values, density_values


def _constant_density_curve(
    value: float,
    points: int,
) -> tuple[list[float], list[float]]:
    """Return a narrow display curve for a sample with no variability.

    Args:
        value: Constant observed value.
        points: Number of x coordinates in the returned curve.

    Returns:
        A tuple containing x values and density values.
    """
    padding: float = max(abs(value) * 0.1, 0.5)
    x_values: list[float] = [
        value - padding + (2.0 * padding * index / (points - 1))
        for index in range(points)
    ]
    density_values: list[float] = [
        math.exp(-0.5 * ((x_value - value) / (padding / 6.0)) ** 2)
        for x_value in x_values
    ]
    return x_values, density_values


def _plot_x_values(values: list[float], points: int) -> list[float]:
    """Return evenly spaced x coordinates that cover the observed values.

    Args:
        values: Theta observations to cover.
        points: Number of x coordinates to return.

    Returns:
        Evenly spaced x coordinates for plotting.
    """
    q1_value: float = percentile(values, 0.25)
    q3_value: float = percentile(values, 0.75)
    interquartile_range: float = q3_value - q1_value
    observed_minimum: float = min(values)
    observed_maximum: float = max(values)
    observed_range: float = observed_maximum - observed_minimum
    padding: float = max(interquartile_range * 0.5, observed_range * 0.1)
    lower_bound: float = observed_minimum - padding
    upper_bound: float = observed_maximum + padding
    return [
        lower_bound + (upper_bound - lower_bound) * index / (points - 1)
        for index in range(points)
    ]


def _pyplot() -> Any:
    """Return matplotlib pyplot configured for non-interactive SVG generation.

    Returns:
        The matplotlib pyplot module.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure: Any, output_path: Path) -> None:
    """Save a matplotlib figure as SVG and close it.

    Args:
        figure: Matplotlib figure to save.
        output_path: Destination SVG file.

    Raises:
        OSError: If the SVG file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, format="svg")
    figure.clear()
