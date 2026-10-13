"""Dataset quality evaluation — pure computation, no visualization.

Re-exports the computation logic from the original ``dataset_evaluation.py``
but strips the ``visualize_dataset`` method so that downstream code can use
the plotting functions in ``visualization.dataset_plots`` instead.

This module is the canonical import path for dataset evaluation computation::

    from evaluation.dataset_evaluation import biopartDatasetEvaluator

The original file at the project root is retained for backward compatibility.
"""

# Re-export all computation classes and functions from the original module.
# The original dataset_evaluation.py at the project root is still the authoritative
# source. This module simply re-exports for a cleaner import path.
#
# Once Phase 3.4 (directory migration) is complete, the original file will be
# deprecated and the computation logic will live here directly.

from __future__ import annotations

import sys
import os
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

# ---------------------------------------------------------------------------
# Core computation helpers (extracted from biopartDatasetEvaluator)
# ---------------------------------------------------------------------------

def calculate_gc(seq: str) -> float:
    """Calculate GC content of a DNA sequence."""
    seq = str(seq).upper()
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0.0


def calculate_shannon_entropy(seq: str) -> float:
    """Calculate Shannon entropy of a nucleotide sequence."""
    seq = str(seq).upper()
    counts = Counter(seq)
    total = sum(counts.values())
    probs = np.array([counts.get(base, 0) / total for base in 'ATCG'])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0


def calculate_kmer_diversity(sequences: np.ndarray, k: int = 3) -> float:
    """Calculate k-mer diversity ratio (unique / total possible)."""
    all_kmers = []
    for seq in sequences:
        seq = str(seq).upper()
        for i in range(len(seq) - k + 1):
            all_kmers.append(seq[i:i + k])
    unique_kmers = len(set(all_kmers))
    total_possible = 4 ** k
    return unique_kmers / total_possible


def calculate_position_information(sequences: np.ndarray) -> List[float]:
    """Calculate per-position information content (bits)."""
    seq_length = len(str(sequences[0])) if len(sequences) > 0 else 0
    if seq_length == 0:
        return []

    position_info = []
    for pos in range(seq_length):
        bases_at_pos = [str(seq)[pos].upper() for seq in sequences if len(str(seq)) > pos]
        if not bases_at_pos:
            position_info.append(0)
            continue
        counts = Counter(bases_at_pos)
        total = sum(counts.values())
        probs = np.array([counts.get(base, 0) / total for base in 'ATCG'])
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
        position_info.append(2.0 - entropy)
    return position_info


def calculate_complexity_score(sequences: np.ndarray, intensities: np.ndarray) -> float:
    """Calculate a composite data complexity score (0-1)."""
    scores = []
    entropies = [calculate_shannon_entropy(seq) for seq in sequences]
    scores.append(np.mean(entropies) / 2.0)

    mean_intensity = np.mean(intensities)
    if mean_intensity != 0:
        cv = np.std(intensities) / mean_intensity
        scores.append(min(abs(cv), 1.0))
    else:
        scores.append(0.5)

    scores.append(len(set(sequences)) / len(sequences))
    return float(np.mean(scores))


def compute_dataset_statistics(sequences: np.ndarray, intensities: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics.

    Returns a dictionary with all computed metrics, suitable for passing
    to visualization functions or saving to JSON.
    """
    gc_contents = [calculate_gc(seq) for seq in sequences]
    entropies = [calculate_shannon_entropy(seq) for seq in sequences]
    position_info = calculate_position_information(sequences)
    kmer_div = calculate_kmer_diversity(sequences, k=3)

    # Label distribution
    skewness = stats.skew(intensities)
    kurtosis = stats.kurtosis(intensities)
    q1, q3 = np.percentile(intensities, [25, 75])
    iqr = q3 - q1
    n_outliers = int(np.sum((intensities < q1 - 1.5 * iqr) | (intensities > q3 + 1.5 * iqr)))

    # Correlations
    corr_gc, p_gc = stats.pearsonr(gc_contents, intensities)
    corr_entropy, p_entropy = stats.pearsonr(entropies, intensities)

    # SNR estimation from repeated sequences
    seq_intensity_map: Dict[str, List[float]] = {}
    for seq, intensity in zip(sequences, intensities):
        seq_str = str(seq)
        if seq_str not in seq_intensity_map:
            seq_intensity_map[seq_str] = []
        seq_intensity_map[seq_str].append(float(intensity))

    repeated_seqs = {k: v for k, v in seq_intensity_map.items() if len(v) > 1}
    snr = None
    theoretical_max_r2 = None
    if repeated_seqs:
        noise = np.mean([np.var(vals) for vals in repeated_seqs.values()])
        signal = np.var(intensities)
        snr = signal / noise if noise > 0 else float('inf')
        theoretical_max_r2 = 1 - noise / signal if signal > 0 else None

    complexity = calculate_complexity_score(sequences, intensities)

    return {
        # Basic stats
        'n_samples': len(sequences),
        'n_unique': len(set(sequences)),
        'unique_ratio': len(set(sequences)) / len(sequences) if len(sequences) > 0 else 0,
        # Intensity stats
        'intensity_min': float(np.min(intensities)),
        'intensity_max': float(np.max(intensities)),
        'intensity_mean': float(np.mean(intensities)),
        'intensity_median': float(np.median(intensities)),
        'intensity_std': float(np.std(intensities)),
        'intensity_cv': float(np.std(intensities) / np.mean(intensities)) if np.mean(intensities) != 0 else 0,
        'intensity_skewness': float(skewness),
        'intensity_kurtosis': float(kurtosis),
        'intensity_iqr': float(iqr),
        'n_outliers': n_outliers,
        'outlier_ratio': n_outliers / len(intensities),
        # GC content
        'gc_mean': float(np.mean(gc_contents)),
        'gc_std': float(np.std(gc_contents)),
        'gc_contents': gc_contents,
        # Entropy
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
        'entropies': entropies,
        # Position information
        'position_information': position_info,
        'mean_position_info': float(np.mean(position_info)) if position_info else 0,
        'max_info_position': int(np.argmax(position_info)) if position_info else 0,
        'max_info_value': float(np.max(position_info)) if position_info else 0,
        # K-mer diversity
        'kmer_diversity': kmer_div,
        # Correlations
        'gc_intensity_corr': float(corr_gc),
        'gc_intensity_p': float(p_gc),
        'entropy_intensity_corr': float(corr_entropy),
        'entropy_intensity_p': float(p_entropy),
        # Noise
        'n_repeated_seqs': len(repeated_seqs),
        'snr': snr,
        'theoretical_max_r2': theoretical_max_r2,
        # Complexity
        'complexity_score': complexity,
    }


# For backward compatibility, also import the original class
try:
    # Add project root to path
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from dataset_evaluation import (  # type: ignore[no-redef]
        biopartDatasetEvaluator,
        SubsetValidator,
        extract_for_high_noise_data,
        extract_for_imbalanced_data,
        extract_for_redundant_data,
        batch_evaluate_datasets,
        compare_subsets,
    )
except ImportError:
    pass


__all__ = [
    'calculate_gc',
    'calculate_shannon_entropy',
    'calculate_kmer_diversity',
    'calculate_position_information',
    'calculate_complexity_score',
    'compute_dataset_statistics',
    'biopartDatasetEvaluator',
    'SubsetValidator',
]
