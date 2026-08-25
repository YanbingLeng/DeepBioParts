"""Training history visualization.

Decoupled from training logic: accepts pre-computed metric histories.
Previously embedded in:
- predictor_regression.py (plot_metrics_curves)
- evo/lora_finetune_evo.py (plot_training_history)
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .nature_style import (
    NPG_COLOR_CYCLE,
    nature_ax,
    save_nature_svg,
    set_nature_ai_style,
)

# ---------------------------------------------------------------------------
# Metric-panel presets for the per-fold overlaid-curve plots (4 panels = 2x2).
# The key is the field name in the history dict; the label is the axis/title text.
# ---------------------------------------------------------------------------
NAIVE_REGRESSION_LAYOUT = [
    ("r2", "$R^2$ Score"), ("pearson_r", "Pearson $r$"),
    ("mae", "MAE"), ("mse", "MSE"),
]
NAIVE_CLASSIFICATION_LAYOUT = [
    ("accuracy", "Accuracy"), ("f1", "F1 Score"),
    ("precision", "Precision"), ("recall", "Recall"),
]
EVO_REGRESSION_LAYOUT = [
    ("train_loss", "Train Loss"), ("val_loss", "Val Loss"),
    ("val_r2", "$R^2$ Score"), ("val_pearson", "Pearson $r$"),
]
EVO_CLASSIFICATION_LAYOUT = [
    ("train_loss", "Train Loss"), ("val_loss", "Val Loss"),
    ("accuracy", "Accuracy"), ("f1", "F1 Score"),
]


def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: str,
    task_type: str = "regression",
    figsize: tuple = (18, 10),
    prefix: str = "",
) -> None:
    """Plot training history curves.

    Args:
        history: Dict mapping metric names to lists of per-epoch values.
            Expected keys for regression: train_loss, val_loss, mae, mse,
            pearson_r, r2.
            Expected keys for classification: train_loss, val_loss, accuracy,
            precision, recall, f1.
        output_dir: Directory to save plots.
        task_type: "regression" or "classification".
        figsize: Figure dimensions.
        prefix: Filename prefix.
    """
    set_nature_ai_style()

    if task_type == "regression":
        _plot_regression_history(history, output_dir, figsize, prefix)
    else:
        _plot_classification_history(history, output_dir, figsize, prefix)


def _plot_regression_history(
    history: Dict[str, List[float]],
    output_dir: str,
    figsize: tuple,
    prefix: str,
) -> None:
    """Plot 2x2 regression training history."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Loss
    ax = axes[0, 0]
    nature_ax(ax)
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # MAE & MSE
    ax = axes[0, 1]
    nature_ax(ax)
    if "mae" in history:
        ax.plot(history["mae"], label="MAE", linewidth=1.5)
    if "mse" in history:
        ax.plot(history["mse"], label="MSE", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title("Error Metrics")
    ax.legend()

    # R²
    ax = axes[1, 0]
    nature_ax(ax)
    if "r2" in history:
        ax.plot(history["r2"], label="$R^2$", color="#2171b5", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$R^2$")
    ax.set_title("$R^2$ Score")
    ax.legend()

    # Pearson & Spearman
    ax = axes[1, 1]
    nature_ax(ax)
    if "pearson_r" in history:
        ax.plot(history["pearson_r"], label="Pearson $r$", linewidth=1.5)
    if "spearman_r" in history:
        ax.plot(history["spearman_r"], label="Spearman $\\rho$", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.set_title("Correlation Metrics")
    ax.legend()

    plt.tight_layout()
    fname = f"{prefix}training_history_regression.png" if prefix else "training_history_regression.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_classification_history(
    history: Dict[str, List[float]],
    output_dir: str,
    figsize: tuple,
    prefix: str,
) -> None:
    """Plot 2x2 classification training history."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Loss
    ax = axes[0, 0]
    nature_ax(ax)
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # Accuracy
    ax = axes[0, 1]
    nature_ax(ax)
    if "accuracy" in history:
        ax.plot(history["accuracy"], label="Accuracy", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()

    # Precision & Recall
    ax = axes[1, 0]
    nature_ax(ax)
    if "precision" in history:
        ax.plot(history["precision"], label="Precision", linewidth=1.5)
    if "recall" in history:
        ax.plot(history["recall"], label="Recall", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall")
    ax.legend()

    # F1
    ax = axes[1, 1]
    nature_ax(ax)
    if "f1" in history:
        ax.plot(history["f1"], label="F1", color="#2171b5", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score")
    ax.legend()

    plt.tight_layout()
    fname = f"{prefix}training_history_classification.png" if prefix else "training_history_classification.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fold_metrics(
    fold_metrics: List[Dict[str, List[float]]],
    output_dir: str,
    task_type: str = "regression",
    figsize: tuple = (2.5, 2.0),
) -> None:
    """Plot validation metrics across folds with mean ± std.

    Args:
        fold_metrics: List of per-fold metric histories.
        output_dir: Directory to save plots.
        task_type: "regression" or "classification".
        figsize: Figure dimensions for Nature-style small plots.
    """
    set_nature_ai_style()

    if task_type == "regression":
        metric_keys = ["mae", "mse", "pearson_r", "r2"]
    else:
        metric_keys = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        nature_ax(ax)

        all_values = [fold.get(key, []) for fold in fold_metrics if key in fold]
        if not all_values:
            continue

        max_len = max(len(v) for v in all_values)
        padded = np.array([np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in all_values])

        epochs = np.arange(max_len)
        mean_vals = np.nanmean(padded, axis=0)
        std_vals = np.nanstd(padded, axis=0)

        ax.plot(epochs, mean_vals, linewidth=0.7)
        ax.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
        ax.set_xlabel("Epoch", fontsize=5.5)
        ax.set_ylabel(key, fontsize=5.5)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "validation_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_fold_training_curves(
    fold_histories: List[Dict[str, List[float]]],
    savepath: str,
    metric_layout: List[tuple],
    filename: str = "training_curves.svg",
) -> None:
    """Overlay one colored curve per fold on each metric panel (Nature-style 2x2 SVG).

    Args:
        fold_histories: list with one ``{metric_key: [per-epoch values]}`` dict per fold.
        savepath: output directory (full path).
        metric_layout: length-4 ``[(metric_key, axis_label), ...]`` filling the
            2x2 panels in order. The presets defined in this module can be
            reused directly: ``NAIVE_REGRESSION_LAYOUT`` /
            ``NAIVE_CLASSIFICATION_LAYOUT`` / ``EVO_REGRESSION_LAYOUT`` /
            ``EVO_CLASSIFICATION_LAYOUT``.
        filename: output SVG filename.
    """
    from matplotlib.ticker import MultipleLocator

    set_nature_ai_style()
    n_folds = len(fold_histories)
    colors = [NPG_COLOR_CYCLE[i % len(NPG_COLOR_CYCLE)] for i in range(n_folds)]

    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    axes = axes.ravel()

    for ax_idx, (metric_key, metric_name) in enumerate(metric_layout):
        ax = axes[ax_idx]
        nature_ax(ax)
        for fold_idx, history in enumerate(fold_histories, 1):
            values = history.get(metric_key, [])
            if values:
                epochs = range(len(values))
                ax.plot(epochs, values, linewidth=0.7, color=colors[fold_idx - 1],
                        label=f"Fold {fold_idx}", alpha=0.8)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        if n_folds <= 5:
            ax.legend(fontsize=6)

    plt.tight_layout()
    output_path = os.path.join(savepath, filename)
    save_nature_svg(fig, output_path)
    plt.close(fig)
    print(f"Training curves saved to: {output_path}")
