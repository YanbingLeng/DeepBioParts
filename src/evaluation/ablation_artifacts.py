"""Ablation-run artifact writing and ablation assembly helpers (library layer).

This module handles standardized artifact writing for ablation runs:
- fold_metrics.csv / run_config.json / parameter_counts.json / test_results.csv
- ablation-mode detection, variant tags, and seed-stable output directory
  construction

Design principle: all functions take explicit parameters (or an already-built
config dict) and never depend on the argparse args namespace, so they stay
decoupled from the CLI and reusable from other entry points (e.g. schedulers,
unit tests).
"""

from __future__ import annotations

import csv
import json
import os
from typing import Iterable, Optional

from src.data.ablation_split import REGRESSION_PASSTHROUGH_COLS

__all__ = [
    "is_ablation",
    "variant_tag",
    "ablation_run_dir",
    "write_fold_metrics",
    "write_run_config",
    "write_parameter_counts",
    "write_ablation_test_results",
]


# ---------------------------------------------------------------------------
# Ablation assembly helpers
# ---------------------------------------------------------------------------

def is_ablation(ablation_variant: Optional[str],
                evo_adaptation: Optional[str],
                pooling_mode: Optional[str]) -> bool:
    """Any ablation flag set counts as ablation mode (enables the fixed external test set)."""
    return (ablation_variant is not None
            or evo_adaptation is not None
            or pooling_mode is not None)


def variant_tag(model_type: str,
                ablation_variant: Optional[str],
                evo_adaptation: Optional[str],
                pooling_mode: Optional[str]) -> str:
    """Variant label used for the output directory."""
    if model_type == "evo":
        return f"adapt-{evo_adaptation or 'lora'}_pool-{pooling_mode or 'attention'}"
    return ablation_variant or "full"


def ablation_run_dir(biopart: str, task_type: str, vtag: str,
                     timestamp: str, output_root: str) -> str:
    """Seed-stable run directory: <root>/<biopart>_<task>/<variant_tag>_<timestamp>/."""
    subdir = f"{biopart}_{task_type}"
    return os.path.join(output_root, subdir, f"{vtag}_{timestamp}")


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def write_fold_metrics(run_dir: str, fold_val_scores: Iterable[float],
                       task_type: str, metric_name: Optional[str] = None) -> None:
    """fold_metrics.csv: monitoring metric per fold.

    Args:
        metric_name: column-name override. Defaults by task_type to
            ``val_pearson_r`` (regression) / ``val_accuracy`` (classification).
            Evo classification ablations use ``val_f1``; pass it explicitly.
    """
    if metric_name is None:
        metric_name = "val_pearson_r" if task_type == "regression" else "val_accuracy"
    path = os.path.join(run_dir, "fold_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", metric_name])
        for i, s in enumerate(fold_val_scores, 1):
            w.writerow([i, f"{float(s):.6f}"])
    print(f"Fold metrics saved to: {path}")


def write_run_config(run_dir: str, config: dict) -> None:
    """run_config.json: full ablation configuration (compatible with existing
    checkpoints; also supports the auto-resume structure).

    Args:
        config: an already-built config dict (the caller fills in fields such
            as variant_tag and n_folds).
    """
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to: {path}")


def write_parameter_counts(run_dir: str, counts: dict) -> None:
    """parameter_counts.json."""
    path = os.path.join(run_dir, "parameter_counts.json")
    with open(path, "w") as f:
        json.dump(counts, f, indent=2)
    print(f"Parameter counts saved to: {path}")


def write_ablation_test_results(run_dir: str, meta_df: pd.DataFrame,
                                test_seqs, y_true, per_fold_preds,
                                ensemble_preds, task_type: str) -> None:
    """Canonical test_results.csv: sample_id, sequence, y_true,
    ensemble_prediction, per-fold predictions + similarity passthrough columns.

    The similarity passthrough column names reuse
    src.data.ablation_split.REGRESSION_PASSTHROUGH_COLS to avoid hard-coding
    the column names again here.
    """
    n_folds = len(per_fold_preds)
    sample_ids = meta_df["sample_id"].values
    path = os.path.join(run_dir, "test_results.csv")
    passthrough_cols = [c for c in REGRESSION_PASSTHROUGH_COLS if c in meta_df.columns]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = (["sample_id", "sequence", "y_true", "ensemble_prediction"]
                  + [f"Fold_{i}_Prediction" for i in range(1, n_folds + 1)]
                  + passthrough_cols)
        w.writerow(header)
        for i in range(len(sample_ids)):
            row = [int(sample_ids[i]), test_seqs[i],
                   f"{y_true[i]:.6f}" if task_type == "regression" else int(y_true[i]),
                   f"{ensemble_preds[i]:.6f}"]
            for k in range(n_folds):
                row.append(f"{per_fold_preds[k][i]:.6f}")
            for c in passthrough_cols:
                row.append(meta_df[c].values[i])
            w.writerow(row)
    print(f"Ablation test_results saved to: {path}")
