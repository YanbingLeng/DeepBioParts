#!/usr/bin/env python3
"""
Non-Repetitive Promoter (NRP) Analysis -- Computation Functions
===============================================================

Pure computation utilities for evaluating non-repetitive promoter sequence
quality via Monte Carlo subsampling, K-mer entropy, and pairwise edit distance
analysis.  All plotting code is intentionally excluded; functions return
structured data (arrays / dicts) that downstream callers can pass to their
own visualisation routines.

Functions
---------
- revcomp(seq) -> str
- canonical_kmer(kmer) -> str
- compute_non_repetitive_count(sequences, lmax) -> (int, int)
- monte_carlo_subsampling(...) -> np.ndarray
- compute_kmer_entropy(sequences, k) -> float
- compute_pairwise_edit_distances(sequences, ...) -> list[int]
"""

import logging
import random
from collections import Counter, defaultdict
from itertools import combinations
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import Levenshtein

logger = logging.getLogger(__name__)

# ============================================================
# Reverse-complement helpers
# ============================================================

_COMPLEMENT = str.maketrans("ATGCUatgcu", "TACGAtacga")


def revcomp(seq: str) -> str:
    """Return the reverse complement of a nucleotide sequence.

    Supports DNA (A/T/G/C) and RNA (U) characters in both upper and lower
    case.  Any character not in the translation table is left unchanged.

    Parameters
    ----------
    seq : str
        Input nucleotide sequence.

    Returns
    -------
    str
        Reverse complement of *seq*.
    """
    return seq.translate(_COMPLEMENT)[::-1]


def canonical_kmer(kmer: str) -> str:
    """Return the lexicographically smaller of a k-mer and its reverse complement.

    This normalises k-mers so that a sequence and its reverse complement are
    treated as identical, which is standard practice for double-stranded DNA
    analyses.

    Parameters
    ----------
    kmer : str
        A short nucleotide string.

    Returns
    -------
    str
        ``min(kmer, revcomp(kmer))``.
    """
    rc = revcomp(kmer)
    return min(kmer, rc)


# ============================================================
# Non-repetitive count (graph-based)
# ============================================================

def compute_non_repetitive_count(
    sequences: List[str],
    lmax: int,
) -> Tuple[int, int]:
    """Compute the size of the largest lmax-non-repetitive subset.

    Two sequences conflict when they share at least one canonical *lmax*-mer.
    Internally, sequences that contain the same *lmax*-mer more than once are
    excluded first.  The remaining sequences form a conflict graph from which
    a maximal independent set is extracted using Pendant Elimination followed
    by a greedy (minimum-degree) strategy.

    Parameters
    ----------
    sequences : list[str]
        Collection of promoter sequences.
    lmax : int
        K-mer length used to define "repetitive".

    Returns
    -------
    tuple[int, int]
        ``(non_repetitive_count, total_sequence_count)`` where *total_sequence_count*
        is simply ``len(sequences)``.
    """
    k = lmax
    n = len(sequences)

    # Step 1: build k-mer sets per sequence and discard those with internal repeats
    seq_kmers: dict[int, set[str]] = {}
    valid_indices: list[int] = []
    for pos in range(n):
        seq = sequences[pos].upper()
        if len(seq) < k:
            continue
        kmers: list[str] = []
        for i in range(len(seq) - k + 1):
            pattern = seq[i:i + k]
            if all(c in "ATCG" for c in pattern):
                kmers.append(canonical_kmer(pattern))

        kmer_counts = Counter(kmers)
        if not any(v >= 2 for v in kmer_counts.values()):
            seq_kmers[pos] = set(kmers)
            valid_indices.append(pos)

    if not valid_indices:
        return 0, n

    # Step 2: inverted index -> conflict graph
    kmer_to_seqs: dict[str, set[int]] = defaultdict(set)
    for pos in valid_indices:
        for kmer in seq_kmers[pos]:
            kmer_to_seqs[kmer].add(pos)

    conflicts: dict[int, set[int]] = defaultdict(set)
    for seq_set in kmer_to_seqs.values():
        seq_list = list(seq_set)
        for i in range(len(seq_list)):
            for j in range(i + 1, len(seq_list)):
                conflicts[seq_list[i]].add(seq_list[j])
                conflicts[seq_list[j]].add(seq_list[i])

    # Step 3: Pendant Elimination + greedy maximal independent set
    remaining: set[int] = set(valid_indices)
    independent_set: set[int] = set()

    while remaining:
        # Pendant Elimination pass
        pendant_found = True
        while pendant_found:
            pendant_found = False
            for node in list(remaining):
                if len(conflicts[node] & remaining) <= 1:
                    independent_set.add(node)
                    remaining.remove(node)
                    remaining -= conflicts[node] & remaining
                    pendant_found = True
                    break

        if not remaining:
            break

        # Greedy: pick the node with fewest remaining conflicts
        node = min(remaining, key=lambda x: len(conflicts[x] & remaining))
        independent_set.add(node)
        remaining.remove(node)
        remaining -= conflicts[node] & remaining

    return len(independent_set), n


# ============================================================
# Monte Carlo subsampling
# ============================================================

def _mc_single_trial(args: Tuple[List[str], int, int, int]) -> float:
    """Execute a single Monte Carlo trial (designed for multiprocessing).

    Parameters
    ----------
    args : tuple
        ``(sequences, lmax, n_sample, seed)``

    Returns
    -------
    float
        Non-repetitive retention rate for this trial.
    """
    sequences_list, lmax, n_sample, seed = args
    rng = random.Random(seed)
    sampled = rng.sample(sequences_list, n_sample)
    n_retained, n_total = compute_non_repetitive_count(sampled, lmax)
    return n_retained / n_total


def monte_carlo_subsampling(
    native_sequences: List[str],
    n_sample: int,
    n_iterations: int,
    lmax: int,
    n_workers: int,
    random_seed: int = 42,
) -> np.ndarray:
    """Monte Carlo subsampling to estimate a baseline non-repetitive retention rate.

    In each iteration, *n_sample* sequences are drawn without replacement from
    *native_sequences*.  The non-repetitive retention rate is computed for the
    sample, and the distribution of rates across all iterations is returned.

    Progress is reported via ``logging`` (level INFO) and a ``tqdm`` progress
    bar -- **no** ``print()`` calls are made.

    Parameters
    ----------
    native_sequences : list[str]
        The full pool of native (natural) sequences.
    n_sample : int
        Number of sequences to draw per iteration.
    n_iterations : int
        Total number of Monte Carlo iterations.
    lmax : int
        K-mer length defining "repetitive".
    n_workers : int
        Number of multiprocessing workers.
    random_seed : int, optional
        Base random seed (each iteration uses ``random_seed + i``).

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(n_iterations,)`` with retention rates in [0, 1].
    """
    logger.info(
        "Monte Carlo subsampling: native_pool=%d, n_sample=%d, "
        "n_iterations=%d, lmax=%d, n_workers=%d",
        len(native_sequences), n_sample, n_iterations, lmax, n_workers,
    )

    seeds = [random_seed + i for i in range(n_iterations)]
    args_list = [
        (native_sequences, lmax, n_sample, s) for s in seeds
    ]

    logger.info("Starting %d Monte Carlo iterations ...", n_iterations)
    with Pool(n_workers) as pool:
        retention_rates = list(
            tqdm(
                pool.imap(_mc_single_trial, args_list),
                total=n_iterations,
                desc="MC iterations",
                ncols=80,
            )
        )

    retention_rates = np.array(retention_rates)
    logger.info(
        "Monte Carlo complete: mean=%.4f, 95%% CI=[%.4f, %.4f]",
        float(np.mean(retention_rates)),
        float(np.percentile(retention_rates, 2.5)),
        float(np.percentile(retention_rates, 97.5)),
    )
    return retention_rates


# ============================================================
# K-mer Shannon entropy
# ============================================================

def compute_kmer_entropy(sequences: List[str], k: int) -> float:
    """Compute Shannon entropy of k-mer frequencies across all sequences.

    ``H = -sum( p_i * log2(p_i) )`` where *p_i* is the observed frequency of
    the *i*-th distinct k-mer.  Only k-mers composed entirely of A/T/C/G
    (after upper-casing) are counted.

    Parameters
    ----------
    sequences : list[str]
        Collection of nucleotide sequences.
    k : int
        K-mer size.

    Returns
    -------
    float
        Shannon entropy in bits.  Returns 0.0 when no valid k-mers are found.
    """
    kmer_counts: Counter = Counter()
    for seq in sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if all(c in "ATCG" for c in kmer):
                kmer_counts[kmer] += 1

    total = sum(kmer_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in kmer_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


# ============================================================
# Pairwise edit distances
# ============================================================

def compute_pairwise_edit_distances(
    sequences: List[str],
    max_pairs: int = 50000,
    random_seed: int = 42,
) -> List[int]:
    """Compute pairwise Levenshtein edit distances among sequences.

    When the number of unique pairs exceeds *max_pairs*, a random subset of
    pairs is evaluated instead of the full all-vs-all matrix.

    Parameters
    ----------
    sequences : list[str]
        Collection of sequences to compare.
    max_pairs : int, optional
        Maximum number of pairs to evaluate (default 50 000).
    random_seed : int, optional
        Seed for the random pair sampler.

    Returns
    -------
    list[int]
        Edit distances for the sampled (or complete) set of pairs.
    """
    n = len(sequences)
    n_pairs = n * (n - 1) // 2

    rng = random.Random(random_seed)

    if n_pairs > max_pairs:
        pairs: list[tuple[int, int]] = []
        indices = list(range(n))
        seen: set[tuple[int, int]] = set()
        while len(pairs) < max_pairs:
            i, j = rng.sample(indices, 2)
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    else:
        pairs = list(combinations(range(n), 2))

    distances: list[int] = []
    for i, j in tqdm(pairs, desc="Computing edit distances", ncols=80, leave=False):
        d = Levenshtein.distance(sequences[i], sequences[j])
        distances.append(d)

    return distances
