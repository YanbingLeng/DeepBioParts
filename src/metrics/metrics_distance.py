"""Sequence distance metrics: Levenshtein, Hamming, etc.

Consolidates distance computations from legacy modules and ``utils/function.py``.
"""

from typing import List, Optional

import numpy as np


def levenshtein_distance(
    seq1: str,
    seq2: str,
) -> int:
    """Compute Levenshtein (edit) distance between two sequences.

    Args:
        seq1: First sequence string.
        seq2: Second sequence string.

    Returns:
        Minimum number of insertions, deletions, and substitutions.
    """
    m, n = len(seq1), len(seq2)

    # Quick check for empty strings
    if m == 0:
        return n
    if n == 0:
        return m

    # Use single-row DP for memory efficiency
    prev_row = list(range(n + 1))

    for i in range(1, m + 1):
        curr_row = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,       # insertion
                prev_row[j] + 1,           # deletion
                prev_row[j - 1] + cost,    # substitution
            )
        prev_row = curr_row

    return prev_row[n]


def hamming_distance(
    seq1: str,
    seq2: str,
) -> int:
    """Compute Hamming distance between two equal-length sequences.

    Args:
        seq1: First sequence string.
        seq2: Second sequence string.

    Returns:
        Number of positions at which sequences differ.

    Raises:
        ValueError: If sequences have different lengths.
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Hamming distance requires equal-length sequences, "
            f"got {len(seq1)} and {len(seq2)}"
        )
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def compute_min_distances_to_reference(
    generated_seqs: List[str],
    reference_seqs: List[str],
    max_reference: int = 5000,
) -> np.ndarray:
    """Compute minimum Levenshtein distance from each generated sequence
    to the nearest reference sequence.

    Args:
        generated_seqs: Sequences to evaluate.
        reference_seqs: Reference (e.g., training) sequences.
        max_reference: Cap on reference sequences for efficiency.

    Returns:
        Array of minimum distances, shape (len(generated_seqs),).
    """
    ref_subset = reference_seqs[:max_reference]
    min_distances = np.zeros(len(generated_seqs), dtype=np.float64)

    for i, gen_seq in enumerate(generated_seqs):
        distances = [levenshtein_distance(gen_seq, ref_seq) for ref_seq in ref_subset]
        min_distances[i] = min(distances)

    return min_distances


def compute_pairwise_distances(
    sequences: List[str],
) -> np.ndarray:
    """Compute pairwise Levenshtein distance matrix.

    Args:
        sequences: List of sequence strings.

    Returns:
        Symmetric distance matrix of shape (N, N).
    """
    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = levenshtein_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def compute_distance_statistics(
    distances: np.ndarray,
) -> dict:
    """Compute summary statistics for a distance array.

    Args:
        distances: Array of distance values.

    Returns:
        Dictionary with mean, median, std, min, max, and percentile stats.
    """
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "p5": float(np.percentile(distances, 5)),
        "p25": float(np.percentile(distances, 25)),
        "p75": float(np.percentile(distances, 75)),
        "p95": float(np.percentile(distances, 95)),
    }
