"""Metrics primitives: regression/classification/k-mer/distance metrics shared by training and evaluation.

Split out from the former ``src/core/`` (the three metrics modules live here;
config and constants have moved to ``src/config/``).
"""

from .metrics import (
    compute_regression_metrics,
    compute_r2_by_bins,
    compute_residuals,
    compute_classification_metrics,
    weighted_spearman,
    ordinal_accuracy,
)
from .metrics_kmer import (
    compute_kmer_frequencies,
    compute_kmer_vector,
    kmer_correlation,
    kmer_js_divergence,
    compute_sequence_diversity,
)
from .metrics_distance import (
    levenshtein_distance,
    hamming_distance,
    compute_min_distances_to_reference,
    compute_pairwise_distances,
    compute_distance_statistics,
)

__all__ = [
    # metrics.py
    "compute_regression_metrics",
    "compute_r2_by_bins",
    "compute_residuals",
    "compute_classification_metrics",
    "weighted_spearman",
    "ordinal_accuracy",
    # metrics_kmer.py
    "compute_kmer_frequencies",
    "compute_kmer_vector",
    "kmer_correlation",
    "kmer_js_divergence",
    "compute_sequence_diversity",
    # metrics_distance.py
    "levenshtein_distance",
    "hamming_distance",
    "compute_min_distances_to_reference",
    "compute_pairwise_distances",
    "compute_distance_statistics",
]
