"""Regression evaluation plots: predicted-vs-actual, residuals, error.

Decoupled from computation: these functions accept pre-computed arrays
and metrics dicts. No model inference or metric calculation happens here.

Previously embedded in:
- train_predictor_regression.py (archived) (L632-689)
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from .nature_style import (
    NATURE_BLUE,
    NATURE_GREEN,
    NATURE_LIGHT_BLUE,
    NATURE_RED,
    SCATTER_ALPHA,
    SCATTER_EDGE,
    SCATTER_SIZE,
    nature_ax,
    save_nature_svg,
    set_nature_ai_style,
)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    save_path: str,
    figsize: tuple = (4 / 2.54, 4 / 2.54),
) -> None:
    """Plot predicted vs actual values with y=x reference line.

    Args:
        y_true: Measured/ground truth values.
        y_pred: Predicted values.
        metrics: Pre-computed metrics dict (needs r2, pearson_r, mae).
        save_path: Output SVG file path.
        figsize: Figure dimensions in inches.
    """
    set_nature_ai_style()

    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=figsize)
    nature_ax(ax)

    ax.scatter(
        y_true, y_pred,
        s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
        color=NATURE_BLUE, edgecolors=SCATTER_EDGE,
    )

    # y = x reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        color=NATURE_RED, linestyle="--", linewidth=0.7, label="y = x",
    )
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    ax.set_xlabel("Measured activity")
    ax.set_ylabel("Predicted activity")

    # Metrics text box
    r2 = metrics.get("r2", 0.0)
    pearson_r = metrics.get("pearson_r", 0.0)
    mae = metrics.get("mae", 0.0)
    textstr = f"$R^2$ = {r2:.3f}\n$r$ = {pearson_r:.3f}\nMAE = {mae:.3f}"
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes, fontsize=7,
        verticalalignment="top", family="sans-serif",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#cccccc",
            linewidth=0.3,
        ),
    )
    ax.legend(loc="lower right")

    save_nature_svg(fig, save_path)


def plot_residuals(
    y_true: np.ndarray,
    residuals: np.ndarray,
    save_path: str,
    figsize: tuple = (4 / 2.54, 4 / 2.54),
) -> None:
    """Plot residuals vs measured values.

    Args:
        y_true: Measured/ground truth values.
        residuals: Pre-computed residuals (y_pred - y_true).
        save_path: Output SVG file path.
        figsize: Figure dimensions in inches.
    """
    set_nature_ai_style()

    fig, ax = plt.subplots(figsize=figsize)
    nature_ax(ax)

    ax.scatter(
        y_true, residuals,
        s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
        color=NATURE_BLUE, edgecolors=SCATTER_EDGE,
    )
    ax.axhline(y=0, color=NATURE_RED, linestyle="--", linewidth=0.7)

    ax.set_xlabel("Measured activity")
    ax.set_ylabel("Residual (predicted \u2212 measured)")

    save_nature_svg(fig, save_path)


def plot_residual_distribution(
    residuals: np.ndarray,
    save_path: str,
    bins: int = 40,
    figsize: tuple = (4 / 2.54, 4 / 2.54),
) -> None:
    """Plot histogram of residual distribution.

    Args:
        residuals: Pre-computed residuals.
        save_path: Output SVG file path.
        bins: Number of histogram bins.
        figsize: Figure dimensions in inches.
    """
    set_nature_ai_style()

    fig, ax = plt.subplots(figsize=figsize)
    nature_ax(ax)

    ax.hist(
        residuals, bins=bins,
        color=NATURE_LIGHT_BLUE, edgecolor=NATURE_BLUE,
        linewidth=0.3, alpha=0.85,
    )
    ax.axvline(x=0, color=NATURE_RED, linestyle="--", linewidth=0.7, label="Zero")
    ax.axvline(
        x=np.mean(residuals), color=NATURE_GREEN, linestyle="-", linewidth=0.7,
        label=f"Mean = {np.mean(residuals):.3f}",
    )

    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.legend()

    save_nature_svg(fig, save_path)


def plot_absolute_error(
    y_true: np.ndarray,
    residuals: np.ndarray,
    save_path: str,
    figsize: tuple = (4 / 2.54, 4 / 2.54),
) -> None:
    """Plot absolute error vs measured values.

    Args:
        y_true: Measured/ground truth values.
        residuals: Pre-computed residuals.
        save_path: Output SVG file path.
        figsize: Figure dimensions in inches.
    """
    set_nature_ai_style()

    fig, ax = plt.subplots(figsize=figsize)
    nature_ax(ax)

    ax.scatter(
        y_true, np.abs(residuals),
        s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
        color=NATURE_BLUE, edgecolors=SCATTER_EDGE,
    )

    ax.set_xlabel("Measured activity")
    ax.set_ylabel("Absolute error")

    save_nature_svg(fig, save_path)


def plot_all_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    output_dir: str,
) -> None:
    """Generate all regression diagnostic plots.

    Convenience function that calls all four plot functions above.

    Args:
        y_true: Measured/ground truth values.
        y_pred: Predicted values.
        metrics: Pre-computed metrics dict.
        output_dir: Directory to save SVG files.
    """
    import os

    residuals = y_pred - y_true

    plot_predicted_vs_actual(y_true, y_pred, metrics, os.path.join(output_dir, "fig_predicted_vs_actual.svg"))
    plot_residuals(y_true, residuals, os.path.join(output_dir, "fig_residuals.svg"))
    plot_residual_distribution(residuals, os.path.join(output_dir, "fig_residual_distribution.svg"))
    plot_absolute_error(y_true, residuals, os.path.join(output_dir, "fig_absolute_error.svg"))


# ---------------------------------------------------------------------------
# NPG Nature palette for Figure 3 dual comparison
# ---------------------------------------------------------------------------
_NPG_DEEP_BLUE = "#3C5488"
_NPG_VERMILLION = "#E64B35"


def plot_dual_scatter_comparison(
    y_true_evo: np.ndarray,
    y_pred_evo: np.ndarray,
    y_true_dl: np.ndarray,
    y_pred_dl: np.ndarray,
    save_path: str,
    label_evo: str = "Evo 1.5 + LoRA",
    label_dl: str = "CNN–Attention–BiLSTM",
    figsize: Tuple[float, float] = (8 / 2.54, 4 / 2.54),
) -> Dict[str, Tuple[float, float]]:
    """Figure 3: side-by-side scatter plots comparing two predictors.

    Args:
        y_true_evo: Ground truth for Evo model predictions.
        y_pred_evo: Evo model predictions.
        y_true_dl: Ground truth for DL model predictions.
        y_pred_dl: DL model predictions.
        save_path: Output SVG/PNG path.
        label_evo: Display name for left panel.
        label_dl: Display name for right panel.
        figsize: Figure dimensions in inches (2 × 4cm square panels).

    Returns:
        Dict with keys 'evo' and 'dl', each mapping to (pearson_r, p_value).
    """
    set_nature_ai_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    datasets = [
        (axes[0], y_true_evo, y_pred_evo, label_evo, _NPG_DEEP_BLUE, "a"),
        (axes[1], y_true_dl, y_pred_dl, label_dl, _NPG_VERMILLION, "b"),
    ]

    results = {}
    for ax, y_true, y_pred, label, color, panel in datasets:
        nature_ax(ax)

        # Scatter
        ax.scatter(
            y_true, y_pred,
            s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
            color=color, edgecolors=SCATTER_EDGE, rasterized=True,
        )

        # y = x reference dashed line
        all_vals = np.concatenate([y_true, y_pred])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.05
        ax.plot(
            [vmin - margin, vmax + margin],
            [vmin - margin, vmax + margin],
            color="#999999", linestyle="--", linewidth=0.7,
        )
        ax.set_xlim(vmin - margin, vmax + margin)
        ax.set_ylim(vmin - margin, vmax + margin)

        # Pearson r
        r, p = pearsonr(y_true, y_pred)
        results[panel] = (r, p)

        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        textstr = f"$r$ = {r:.3f}\n{p_str}"
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", family="sans-serif",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white", edgecolor="#cccccc", linewidth=0.3,
            ),
        )

        # Axis labels
        ax.set_xlabel("True Strength")
        ax.set_ylabel("Predicted Strength")

        # Panel label
        ax.text(
            -0.18, 1.08, panel,
            transform=ax.transAxes, fontsize=7,
            fontweight="bold", fontfamily="sans-serif",
        )

        # Subtle model name in top-right
        ax.text(
            0.95, 0.95, label,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="right",
            family="sans-serif", color="#555555",
        )

    fig.tight_layout(w_pad=1.5)
    save_nature_svg(fig, save_path, dpi=600)

    return {"evo": results["a"], "dl": results["b"]}
