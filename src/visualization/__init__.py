"""DeepBioParts visualization package.

All plotting functions follow the principle of computation-display separation:
they accept pre-computed data (Dict, np.ndarray) and produce figures.
They never perform model inference or metric computation.
"""

from .nature_style import (
    set_nature_ai_style,
    setup_nature_style,
    nature_ax,
    save_nature_svg,
    NPG_COLOR_CYCLE,
)
from .regression_plots import plot_predicted_vs_actual, plot_residuals, plot_residual_distribution
from .training_plots import (
    plot_training_history,
    plot_per_fold_training_curves,
    NAIVE_REGRESSION_LAYOUT,
    NAIVE_CLASSIFICATION_LAYOUT,
    EVO_REGRESSION_LAYOUT,
    EVO_CLASSIFICATION_LAYOUT,
)
from .scan_plots import BIOPART_COLORS, BIOPART_LABELS, plot_scan_results
from .dataset_plots import plot_dataset_overview

__all__ = [
    "set_nature_ai_style",
    "setup_nature_style",
    "NPG_COLOR_CYCLE",
    "nature_ax",
    "save_nature_svg",
    "plot_predicted_vs_actual",
    "plot_residuals",
    "plot_residual_distribution",
    "plot_training_history",
    "plot_per_fold_training_curves",
    "NAIVE_REGRESSION_LAYOUT",
    "NAIVE_CLASSIFICATION_LAYOUT",
    "EVO_REGRESSION_LAYOUT",
    "EVO_CLASSIFICATION_LAYOUT",
    "BIOPART_COLORS",
    "BIOPART_LABELS",
    "plot_scan_results",
    "plot_dataset_overview",
]
