"""K-mer frequency metrics for DNA sequence evaluation.

Consolidates k-mer computation logic from legacy modules, ``utils/pc_util.py``,
and ``gpro/evaluator/kmer.py``.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr


def compute_kmer_frequencies(
    sequences: List[str],
    k: int = 4,
) -> Dict[str, float]:
    """Compute normalized k-mer frequency distribution.

    Args:
        sequences: List of DNA sequences.
        k: K-mer size.

    Returns:
        Dictionary mapping k-mer strings to normalized frequencies.
    """
    kmer_counts: Dict[str, int] = {}
    total = 0

    for seq in sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if all(c in "ACGT" for c in kmer):
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
                total += 1

    if total == 0:
        return {}

    return {kmer: count / total for kmer, count in kmer_counts.items()}


def compute_kmer_vector(
    sequence: str,
    k: int = 3,
) -> np.ndarray:
    """Compute a fixed-length k-mer frequency vector for a single sequence.

    Generates all possible k-mers in alphabetical order and returns
    a vector of their frequencies.

    Args:
        sequence: DNA sequence string.
        k: K-mer size.

    Returns:
        Numpy array of shape (4^k,) with k-mer frequencies.
    """
    sequence = sequence.upper()
    bases = ["A", "C", "G", "T"]

    # Generate all possible k-mers in order
    from itertools import product

    all_kmers = ["".join(p) for p in product(bases, repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}

    vector = np.zeros(len(all_kmers), dtype=np.float64)
    count = 0

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        if kmer in kmer_to_idx:
            vector[kmer_to_idx[kmer]] += 1
            count += 1

    if count > 0:
        vector /= count

    return vector


def kmer_correlation(
    seqs_a: List[str],
    seqs_b: List[str],
    k: int = 4,
) -> Dict[str, float]:
    """Compute k-mer frequency correlation between two sequence sets.

    Args:
        seqs_a: First set of sequences (e.g., training data).
        seqs_b: Second set of sequences (e.g., generated data).
        k: K-mer size.

    Returns:
        Dictionary with pearson_r and pearson_p values.
    """
    freq_a = compute_kmer_frequencies(seqs_a, k)
    freq_b = compute_kmer_frequencies(seqs_b, k)

    # Align k-mers (union of both sets)
    all_kmers = sorted(set(freq_a.keys()) | set(freq_b.keys()))

    vec_a = np.array([freq_a.get(km, 0.0) for km in all_kmers])
    vec_b = np.array([freq_b.get(km, 0.0) for km in all_kmers])

    if len(all_kmers) < 2:
        return {"pearson_r": 0.0, "pearson_p": 1.0}

    r, p = pearsonr(vec_a, vec_b)
    return {"pearson_r": float(r), "pearson_p": float(p)}


def kmer_js_divergence(
    freq_p: Dict[str, float],
    freq_q: Dict[str, float],
    epsilon: float = 1e-10,
) -> float:
    """Compute Jensen-Shannon divergence between two k-mer distributions.

    Args:
        freq_p: First k-mer frequency distribution.
        freq_q: Second k-mer frequency distribution.
        epsilon: Smoothing constant to avoid log(0).

    Returns:
        JS divergence value (0 to ln(2)).
    """
    all_kmers = sorted(set(freq_p.keys()) | set(freq_q.keys()))

    p = np.array([freq_p.get(km, 0.0) + epsilon for km in all_kmers])
    q = np.array([freq_q.get(km, 0.0) + epsilon for km in all_kmers])

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return float(0.5 * (kl_pm + kl_qm))


def compute_sequence_diversity(
    sequences: List[str],
) -> float:
    """Compute sequence diversity as fraction of unique sequences.

    Args:
        sequences: List of DNA sequences.

    Returns:
        Diversity score in [0, 1]. 1.0 means all sequences are unique.
    """
    if len(sequences) == 0:
        return 0.0
    return float(len(set(sequences)) / len(sequences))
