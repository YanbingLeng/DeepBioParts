"""Dataset overview plots: activity, GC content, entropy, information, Q-Q.

Decoupled from computation: these functions accept pre-computed arrays
and metric lists. No metric calculation happens here.

Previously embedded in:
- dataset_evaluation.py (biopartDatasetEvaluator.visualize_dataset, L1014-1077)
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from .nature_style import (
    NATURE_BLUISH_GREEN,
    NATURE_BLUE,
    NATURE_GREEN,
    NATURE_SKY_BLUE,
    NATURE_VERMILLION,
    NATURE_YELLOW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
    nature_ax,
    save_nature_svg,
    set_nature_ai_style,
)


def plot_dataset_overview(
    intensities: np.ndarray,
    sequences: np.ndarray,
    gc_contents: Optional[List[float]] = None,
    entropies: Optional[List[float]] = None,
    position_information: Optional[List[float]] = None,
    save_path: str = "dataset_overview.png",
    dpi: int = 150,
) -> None:
    """Create a 2x3 overview figure of dataset characteristics.

    Panels:
        - (0,0) Activity distribution histogram
        - (0,1) GC content distribution histogram
        - (0,2) GC content vs Activity scatter
        - (1,0) Sequence entropy distribution histogram
        - (1,1) Position-wise information content bar chart
        - (1,2) Q-Q plot of activity values

    When optional data is ``None`` the corresponding panel is left empty
    with a "No data" placeholder so the figure layout remains intact.

    Args:
        intensities: 1-D array of activity / expression values.
        sequences: Array-like of nucleotide sequences (used for panel titles
            only; all metrics must be pre-computed and passed separately).
        gc_contents: Pre-computed GC content per sequence.  If ``None`` the
            GC histogram and scatter panels are skipped.
        entropies: Pre-computed Shannon entropy per sequence.  If ``None``
            the entropy histogram panel is skipped.
        position_information: Pre-computed per-position information content
            (bits).  If ``None`` the information bar chart is skipped.
        save_path: File path for the output figure.
        dpi: Resolution in dots per inch for the saved figure.
    """
    set_nature_ai_style()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ---- Panel (0, 0): Activity distribution ----
    ax = axes[0, 0]
    nature_ax(ax)
    ax.hist(intensities, bins=50, edgecolor="black", alpha=0.7, color=NATURE_BLUE)
    ax.set_xlabel("Activity")
    ax.set_ylabel("Count")
    ax.set_title("Activity Distribution")

    # ---- Panel (0, 1): GC content distribution ----
    ax = axes[0, 1]
    nature_ax(ax)
    if gc_contents is not None:
        ax.hist(gc_contents, bins=30, edgecolor="black", alpha=0.7, color=NATURE_GREEN)
        ax.set_xlabel("GC Content")
        ax.set_ylabel("Count")
        ax.set_title("GC Content Distribution")
    else:
        ax.set_title("GC Content Distribution")
        ax.text(
            0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="gray",
        )

    # ---- Panel (0, 2): GC content vs Activity scatter ----
    ax = axes[0, 2]
    nature_ax(ax)
    if gc_contents is not None:
        ax.scatter(
            gc_contents,
            intensities,
            alpha=SCATTER_ALPHA,
            s=SCATTER_SIZE,
            color=NATURE_VERMILLION,
        )
        ax.set_xlabel("GC Content")
        ax.set_ylabel("Activity")
        ax.set_title("GC Content vs Activity")
    else:
        ax.set_title("GC Content vs Activity")
        ax.text(
            0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="gray",
        )

    # ---- Panel (1, 0): Sequence entropy distribution ----
    ax = axes[1, 0]
    nature_ax(ax)
    if entropies is not None:
        ax.hist(
            entropies, bins=30, edgecolor="black", alpha=0.7,
            color=NATURE_BLUISH_GREEN,
        )
        ax.set_xlabel("Sequence Entropy")
        ax.set_ylabel("Count")
        ax.set_title("Sequence Entropy Distribution")
    else:
        ax.set_title("Sequence Entropy Distribution")
        ax.text(
            0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="gray",
        )

    # ---- Panel (1, 1): Position-wise information content ----
    ax = axes[1, 1]
    nature_ax(ax)
    if position_information is not None and len(position_information) > 0:
        ax.bar(
            range(len(position_information)),
            position_information,
            color=NATURE_YELLOW,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("Information (bits)")
        ax.set_title("Position-wise Information Content")
    else:
        ax.set_title("Position-wise Information Content")
        ax.text(
            0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="gray",
        )

    # ---- Panel (1, 2): Q-Q plot of activity ----
    ax = axes[1, 2]
    nature_ax(ax)
    stats.probplot(intensities, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Activity)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
