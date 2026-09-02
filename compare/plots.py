"""SVG plot generation for SafeLLM4SE two-sample comparison reports."""

import inspect
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from compare.models import ComparisonSettings
from reporting.plots import _pyplot, _save_figure, kde_curve

ComparisonPlotFunction = Callable[
    [list[float], list[float], str, str, Path, str],
    None,
]


def generate_requested_comparison_plots(
    values_1: list[float],
    values_2: list[float],
    settings: ComparisonSettings,
) -> list[Path]:
    """Generate every two-sample plot requested in comparison settings.

    Args:
        values_1: Theta observations for the first task.
        values_2: Theta observations for the second task.
        settings: Runtime settings containing optional plot paths.

    Returns:
        Paths of the generated SVG files.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If any SVG file cannot be written.
        ValueError: If theta values are empty.
    """
    if not values_1 or not values_2:
        raise ValueError("Both theta samples must contain at least one value.")
    label_1: str = settings.task_name_1 or settings.task_id_1
    label_2: str = settings.task_name_2 or settings.task_id_2
    title: str = f"Theta distributions: {label_1} vs {label_2}"
    requested_plots: list[tuple[Path | None, ComparisonPlotFunction]] = [
        (settings.boxplot_path, create_comparison_boxplot),
        (settings.violin_path, create_comparison_violin_plot),
        (settings.ecdf_path, create_comparison_ecdf_plot),
        (settings.raincloud_path, create_comparison_raincloud_plot),
        (settings.kde_path, create_comparison_kde_plot),
    ]
    generated_paths: list[Path] = []
    for output_path, plot_function in requested_plots:
        if output_path is None:
            continue
        logger.info("Generating comparison plot {}.", output_path)
        plot_function(values_1, values_2, label_1, label_2, output_path, title)
        generated_paths.append(output_path)
    return generated_paths


def create_comparison_boxplot(
    values_1: list[float],
    values_2: list[float],
    label_1: str,
    label_2: str,
    output_path: Path,
    title: str,
) -> None:
    """Create an SVG boxplot comparing two theta samples.

    Args:
        values_1: Theta observations for the first task.
        values_2: Theta observations for the second task.
        label_1: X-axis label for the first task.
        label_2: X-axis label for the second task.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    boxplot_options: dict[str, list[str]] = _boxplot_tick_label_options(
        axis.boxplot,
        [label_1, label_2],
    )
    boxplot = axis.boxplot(
        [values_1, values_2],
        patch_artist=True,
        medianprops={"color": "#b91c1c", "linewidth": 2.0},
        whiskerprops={"color": "#1f2937"},
        capprops={"color": "#1f2937"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#f97316",
            "markeredgecolor": "#7c2d12",
            "alpha": 0.75,
        },
        **boxplot_options,
    )
    for box_patch, color in zip(boxplot["boxes"], ["#9ecae1", "#a7f3d0"]):
        box_patch.set_facecolor(color)
        box_patch.set_edgecolor("#1f2937")
    axis.set_title(title)
    axis.set_ylabel("theta")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_path)


def create_comparison_violin_plot(
    values_1: list[float],
    values_2: list[float],
    label_1: str,
    label_2: str,
    output_path: Path,
    title: str,
) -> None:
    """Create an SVG violin plot comparing two theta samples.

    Args:
        values_1: Theta observations for the first task.
        values_2: Theta observations for the second task.
        label_1: X-axis label for the first task.
        label_2: X-axis label for the second task.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    violin_parts: dict[str, Any] = axis.violinplot(
        [values_1, values_2],
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for body, color in zip(violin_parts["bodies"], ["#9ecae1", "#a7f3d0"]):
        body.set_facecolor(color)
        body.set_edgecolor("#1f2937")
        body.set_alpha(0.8)
    axis.set_title(title)
    axis.set_ylabel("theta")
    axis.set_xticks([1, 2])
    axis.set_xticklabels([label_1, label_2])
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_path)


def create_comparison_ecdf_plot(
    values_1: list[float],
    values_2: list[float],
    label_1: str,
    label_2: str,
    output_path: Path,
    title: str,
) -> None:
    """Create an SVG ECDF plot comparing two theta samples.

    Args:
        values_1: Theta observations for the first task.
        values_2: Theta observations for the second task.
        label_1: Legend label for the first task.
        label_2: Legend label for the second task.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for values, label, color in [
        (values_1, label_1, "#2563eb"),
        (values_2, label_2, "#059669"),
    ]:
        sorted_values: list[float] = sorted(values)
        probabilities: list[float] = [
            index / len(sorted_values)
            for index in range(1, len(sorted_values) + 1)
        ]
        axis.step(sorted_values, probabilities, where="post", label=label, color=color)
    axis.set_title(title)
    axis.set_xlabel("theta")
    axis.set_ylabel("ECDF")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_path)


def create_comparison_raincloud_plot(
    values_1: list[float],
    values_2: list[float],
    label_1: str,
    label_2: str,
    output_path: Path,
    title: str,
) -> None:
    """Create an SVG raincloud plot comparing two theta samples.

    Args:
        values_1: Theta observations for the first task.
        values_2: Theta observations for the second task.
        label_1: Y-axis label for the first task.
        label_2: Y-axis label for the second task.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    _draw_raincloud_sample(axis, values_1, 1.0, "#2563eb")
    _draw_raincloud_sample(axis, values_2, 0.0, "#059669")
    axis.set_title(title)
    axis.set_xlabel("theta")
    axis.set_yticks([1.0, 0.0])
    axis.set_yticklabels([label_1, label_2])
    axis.grid(axis="x", alpha=0.25)
    _save_figure(figure, output_path)


def create_comparison_kde_plot(
    values_1: list[float],
    values_2: list[float],
    label_1: str,
    label_2: str,
    output_path: Path,
    title: str,
) -> None:
    """Create an SVG KDE plot comparing two theta samples.

    Args:
        values_1: Theta observations for the first task.
        values_2: Theta observations for the second task.
        label_1: Legend label for the first task.
        label_2: Legend label for the second task.
        output_path: SVG file path to write.
        title: Plot title.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If the SVG file cannot be written.
    """
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for values, label, color in [
        (values_1, label_1, "#2563eb"),
        (values_2, label_2, "#059669"),
    ]:
        x_values, density_values = kde_curve(values)
        axis.plot(x_values, density_values, color=color, linewidth=2.0, label=label)
        axis.fill_between(x_values, density_values, color=color, alpha=0.18)
    axis.set_title(title)
    axis.set_xlabel("theta")
    axis.set_ylabel("density")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_path)


def _draw_raincloud_sample(
    axis: Any,
    values: list[float],
    baseline: float,
    color: str,
) -> None:
    """Draw one horizontal raincloud sample on an axis.

    Args:
        axis: Matplotlib axis where the sample is drawn.
        values: Theta observations for one task.
        baseline: Vertical baseline for the sample.
        color: Color used for the density and observations.
    """
    x_values, density_values = kde_curve(values)
    maximum_density: float = max(density_values) if density_values else 1.0
    scaled_density: list[float] = [
        baseline + 0.28 * density_value / maximum_density
        for density_value in density_values
    ]
    scatter_y_values: list[float] = [
        baseline - 0.20 + 0.08 * ((index % 7) / 6.0)
        for index, _ in enumerate(values)
    ]
    axis.fill_between(
        x_values,
        baseline,
        scaled_density,
        color=color,
        alpha=0.25,
        linewidth=0.0,
    )
    axis.plot(x_values, scaled_density, color=color, linewidth=1.5)
    axis.boxplot(
        values,
        vert=False,
        positions=[baseline - 0.08],
        widths=[0.12],
        patch_artist=True,
        boxprops={"facecolor": "#e5e7eb", "edgecolor": "#1f2937"},
        medianprops={"color": "#b91c1c", "linewidth": 2.0},
        whiskerprops={"color": "#1f2937"},
        capprops={"color": "#1f2937"},
    )
    axis.scatter(values, scatter_y_values, s=18, color=color, alpha=0.72)


def _boxplot_tick_label_options(
    boxplot_function: Callable[..., Any],
    labels: list[str],
) -> dict[str, list[str]]:
    """Return the supported Matplotlib boxplot tick-label keyword.

    Args:
        boxplot_function: Matplotlib boxplot function to inspect.
        labels: Labels to display for the boxplot ticks.

    Returns:
        Keyword arguments compatible with the installed Matplotlib version.

    Raises:
        ValueError: If the boxplot function signature cannot be inspected.
    """
    boxplot_signature: inspect.Signature = inspect.signature(boxplot_function)
    if "tick_labels" in boxplot_signature.parameters:
        return {"tick_labels": labels}
    return {"labels": labels}
