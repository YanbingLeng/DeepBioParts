"""Unified regression and classification metrics module.

This module consolidates metric computations that were previously duplicated
across 7+ files (train_predictor_regression.py, predictor_regression.py,
lora_finetune_evo.py, etc.).

All functions are pure: no side effects, no plotting, no I/O.
They accept numpy arrays and return Dict[str, float].
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


# ============================================================================
# Regression Metrics
# ============================================================================


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all standard regression metrics.

    Args:
        y_true: Ground truth values, shape (N,).
        y_pred: Predicted values, shape (N,).

    Returns:
        Dictionary with keys: r2, pearson_r, pearson_p, spearman_r,
        spearman_p, mse, mae, rmse.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    pearson_r_val, pearson_p = pearsonr(y_true, y_pred)
    spearman_r_val, spearman_p = spearmanr(y_true, y_pred)

    return {
        "r2": r2,
        "pearson_r": float(pearson_r_val),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r_val),
        "spearman_p": float(spearman_p),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }


def compute_r2_by_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int = 5,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> Dict[str, float]:
    """Compute R² score within equal-width bins of the value range.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        num_bins: Number of bins.
        min_val: Minimum value for binning range.
        max_val: Maximum value for binning range.

    Returns:
        Dictionary mapping bin label to R² score.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    result = {}

    for i in range(num_bins):
        mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
        if i == num_bins - 1:
            mask = (y_true >= bin_edges[i]) & (y_true <= bin_edges[i + 1])

        if mask.sum() >= 2:
            result[f"bin_{i}_{bin_edges[i]:.2f}_{bin_edges[i+1]:.2f}"] = float(
                r2_score(y_true[mask], y_pred[mask])
            )
        else:
            result[f"bin_{i}_{bin_edges[i]:.2f}_{bin_edges[i+1]:.2f}"] = float("nan")

    return result


def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute residuals and their summary statistics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Tuple of (residuals array, statistics dict).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    residuals = y_pred - y_true

    stats = {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
        "abs_error_mean": float(np.mean(np.abs(residuals))),
        "abs_error_median": float(np.median(np.abs(residuals))),
    }

    return residuals, stats


# ============================================================================
# Classification Metrics
# ============================================================================


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method for precision/recall/f1.

    Returns:
        Dictionary with keys: accuracy, precision, recall, f1.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


# ============================================================================
# Ordinal Regression Metrics
# ============================================================================


def weighted_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute weighted Spearman correlation emphasizing extreme values.

    Weights are proportional to the squared deviation from the mean,
    giving more importance to extreme predictions.

    Args:
        y_true: Ground truth ordinal values.
        y_pred: Predicted ordinal values.

    Returns:
        Weighted Spearman correlation coefficient.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mean_true = np.mean(y_true)
    weights = (y_true - mean_true) ** 2
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

    n = len(y_true)
    weighted_mean_true = np.sum(weights * y_true)
    weighted_mean_pred = np.sum(weights * y_pred)

    cov = np.sum(weights * (y_true - weighted_mean_true) * (y_pred - weighted_mean_pred))
    std_true = np.sqrt(np.sum(weights * (y_true - weighted_mean_true) ** 2))
    std_pred = np.sqrt(np.sum(weights * (y_pred - weighted_mean_pred) ** 2))

    if std_true < 1e-10 or std_pred < 1e-10:
        return 0.0

    return float(cov / (std_true * std_pred))


def ordinal_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 1,
) -> float:
    """Compute accuracy with ordinal tolerance.

    A prediction is considered correct if it is within `tolerance`
    classes of the true label.

    Args:
        y_true: Ground truth ordinal labels.
        y_pred: Predicted ordinal labels.
        tolerance: Allowed deviation in number of classes.

    Returns:
        Accuracy within tolerance.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    correct = np.abs(y_true - y_pred) <= tolerance
    return float(np.mean(correct))
