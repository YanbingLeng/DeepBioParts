"""Data splitting utilities facade module.

Re-exports cluster-based and random data-splitting helpers from ``utils.data``
for backward compatibility during migration.

Functions:
    cluster_based_split: Cluster-based train/test split using k-mer similarity.
    ClusterBasedKFold: Cluster-based K-fold cross-validation iterator.
    cluster_based_k_fold_split: Functional wrapper around ClusterBasedKFold.
    split_data: Random train/validation split for unsupervised data.
    split_data_supervised: Random train/validation split for supervised data.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Cluster-based splitting utilities
# ---------------------------------------------------------------------------

try:
    from utils.data import (  # type: ignore[no-redef]
        cluster_based_split,
        ClusterBasedKFold,
        cluster_based_k_fold_split,
    )
except ImportError:
    cluster_based_split = None  # type: ignore[assignment,misc]
    ClusterBasedKFold = None  # type: ignore[assignment,misc]
    cluster_based_k_fold_split = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Random splitting utilities
# ---------------------------------------------------------------------------

try:
    from utils.data import (  # type: ignore[no-redef]
        split_data,
        split_data_supervised,
    )
except ImportError:
    split_data = None  # type: ignore[assignment,misc]
    split_data_supervised = None  # type: ignore[assignment,misc]

__all__ = [
    "cluster_based_split",
    "ClusterBasedKFold",
    "cluster_based_k_fold_split",
    "split_data",
    "split_data_supervised",
]
