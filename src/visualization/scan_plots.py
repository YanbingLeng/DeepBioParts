"""Scan-result line plots (Nature style).

Sliding-window scan line plots. Follows a compute/presentation split: this
module only receives an already-computed prediction DataFrame and produces
SVG/PNG output; it performs no inference.

Classification-aware: besides the regression column ``predicted_activity``,
the classification columns ``prob_positive`` / ``max_probability`` are also
supported, avoiding a KeyError on classification results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .nature_style import nature_ax, save_nature_svg, set_nature_ai_style

__all__ = ["BIOPART_COLORS", "BIOPART_LABELS", "plot_scan_results"]

# NPG color palette per biopart
BIOPART_COLORS = {
    "rbs":        "#E64B35",   # NPG red
    "promoter":   "#4DBBD5",   # NPG cyan
    "terminator": "#00A087",   # NPG green
}

BIOPART_LABELS = {
    "rbs":        "RBS (rel. RBS30)",
    "promoter":   "Promoter (rel. J23101)",
    "terminator": "Terminator P(strong)",
}

# Candidate y-value columns for the line plot (regression / classification),
# in priority order
_VALUE_COLS = ("predicted_activity", "prob_positive", "max_probability")


def _value_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first plottable value column (regression or classification) in the DataFrame, or None if absent."""
    for c in _VALUE_COLS:
        if c in df.columns:
            return c
    return None


def _ylabel_for(biopart: Optional[str], value_col: str) -> str:
    """Infer the y-axis label from the biopart and value column."""
    if value_col in ("prob_positive", "max_probability"):
        if biopart == "terminator":
            return "P(strong terminator)"
        return "Probability"
    # Regression
    if biopart == "rbs":
        return "Relative activity (rel. RBS30)"
    if biopart == "promoter":
        return "Relative activity (rel. J23101)"
    return "Relative activity"


def plot_scan_results(report_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the scan-prediction line chart.

    - Multi-part mode (``biopart`` column present): promoter/RBS share the left
      axis (relative activity); terminator uses the right axis (strong
      terminator probability).
    - Single-part mode: one y axis; automatically chooses between the
      regression column and a classification probability column.
    - Outputs SVG + PNG (Nature style).
    """
    value_col = _value_column(report_df)
    if value_col is None:
        print("  Skipping plot: no prediction result columns")
        return

    set_nature_ai_style()

    has_multi_bioparts = "biopart" in report_df.columns
    bioparts_present = sorted(report_df["biopart"].unique()) if has_multi_bioparts else [None]

    # --- Estimate figure width from sequence length ---
    if "start" in report_df.columns:
        end_max = report_df["end"].max() if "end" in report_df.columns else report_df["start"].max() + 15
        seq_len = report_df["start"].max() + (end_max - report_df["start"].max())
    else:
        seq_len = 50
    width_mm = min(max(89, seq_len * 0.8), 183 * 1.5)
    fig_width = width_mm / 25.4
    fig_height = 55 / 25.4  # Nature standard height

    fig, ax_left = plt.subplots(figsize=(fig_width, fig_height))

    has_right_axis = False
    ax_right = None

    if has_multi_bioparts:
        # --- Multi-part mode ---
        for bp in ["rbs", "promoter", "terminator"]:
            if bp not in bioparts_present:
                continue
            sub = report_df[report_df["biopart"] == bp].sort_values("start")
            color = BIOPART_COLORS.get(bp, "#333333")

            if bp == "terminator":
                # terminator uses the right axis, showing P(strong terminator)
                if not has_right_axis:
                    ax_right = ax_left.twinx()
                    nature_ax(ax_right, spine_positions=["right"])
                    has_right_axis = True
                y_vals = sub["prob_positive"] if "prob_positive" in sub.columns else sub[_value_column(sub)]
                ax_right.plot(sub["start"], y_vals, color=color, linewidth=1.0,
                              label=BIOPART_LABELS.get(bp, bp))
            else:
                ax_left.plot(sub["start"], sub[value_col], color=color, linewidth=1.0,
                             label=BIOPART_LABELS.get(bp, bp))

        nature_ax(ax_left, spine_positions=["left", "bottom"])
        ax_left.set_xlabel("Position (bp)")
        ax_left.set_ylabel("Relative activity")
        if has_right_axis and ax_right is not None:
            ax_right.set_ylabel("P(strong terminator)")
            ax_right.set_ylim(0, 1)

        lines_left, labels_left = ax_left.get_legend_handles_labels()
        if has_right_axis and ax_right is not None:
            lines_right, labels_right = ax_right.get_legend_handles_labels()
        else:
            lines_right, labels_right = [], []
        ax_left.legend(lines_left + lines_right, labels_left + labels_right,
                       loc="upper right", frameon=True)
    else:
        # --- Single-part mode ---
        if "start" in report_df.columns:
            sub = report_df.sort_values("start")
            x = sub["start"]
        else:
            x = np.arange(len(report_df))
            sub = report_df

        bp_hint = bioparts_present[0] if (has_multi_bioparts and len(bioparts_present) == 1) else None
        color = BIOPART_COLORS.get(bp_hint, "#2171B5")

        ax_left.plot(x, sub[value_col], color=color, linewidth=1.0)
        nature_ax(ax_left, spine_positions=["left", "bottom"])
        ax_left.set_xlabel("Position (bp)")
        ax_left.set_ylabel(_ylabel_for(bp_hint, value_col))

    fig_path = str(output_dir / "scan_plot")
    save_nature_svg(fig, fig_path)
    plt.close(fig)
    print(f"Scan line plot saved: {fig_path}.svg / .png")
