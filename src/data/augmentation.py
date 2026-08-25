"""DNA sequence data augmentation: random mutation and Mixup.

Provides two augmentation strategies:
- random_mutate_onehot: random base mutations on one-hot encodings, suited for sparse datasets (e.g. terminator)
- mixup_batch: batch-level Mixup regularization at training time, suited for rugged landscapes (e.g. RBS)
"""

import torch
import numpy as np


def random_mutate_onehot(
    features: torch.Tensor,
    n_mutations: int = 1,
) -> torch.Tensor:
    """Apply random single-base mutations to one-hot encoded sequences.

    For each sequence, randomly selects ``n_mutations`` positions and replaces
    the base at each position with a randomly chosen different base (excluding
    the original), generating virtual neighbor samples.

    Args:
        features: one-hot encoded tensor, shape (N, seq_len, 4)
        n_mutations: number of positions to mutate per sequence

    Returns:
        Mutated tensor, shape (N, seq_len, 4)
    """
    N, L, C = features.shape
    mutated = features.clone()

    for i in range(N):
        # Randomly choose mutation positions
        positions = np.random.choice(L, size=min(n_mutations, L), replace=False)
        for pos in positions:
            # Find the current base index
            orig_base = features[i, pos].argmax().item()
            # Randomly pick a different base
            new_base = np.random.choice([b for b in range(4) if b != orig_base])
            mutated[i, pos] = 0.0
            mutated[i, pos, new_base] = 1.0

    return mutated


def mixup_batch(
    feature: torch.Tensor,
    label: torch.Tensor,
    alpha: float = 0.2,
) -> tuple:
    """Apply Mixup augmentation to a batch.

    Randomly pairs samples within the batch and forms convex combinations:
        feature_new = lambda * feature_i + (1 - lambda) * feature_j
        label_new   = lambda * label_i   + (1 - lambda) * label_j
    where lambda ~ Beta(alpha, alpha).

    Args:
        feature: input feature tensor, shape (B, ...)
        label: label tensor, shape (B,)
        alpha: Beta distribution parameter; smaller values reduce mixing

    Returns:
        Tuple (mixed_feature, mixed_label)
    """
    if alpha <= 0:
        return feature, label

    lam = np.random.beta(alpha, alpha)
    # Bias lambda toward the original sample (keep at least 50% of the original signal)
    lam = max(lam, 1 - lam)

    perm = torch.randperm(feature.size(0), device=feature.device)
    mixed_feature = lam * feature + (1 - lam) * feature[perm]
    mixed_label = lam * label + (1 - lam) * label[perm]

    return mixed_feature, mixed_label


def neighbor_interpolate(
    features: torch.Tensor,
    labels: torch.Tensor,
    seqs: list,
    lambdas: list = None,
    max_hamming: int = 1,
) -> tuple:
    """Linearly interpolate between Hamming-neighbor pairs to generate virtual samples.

    On the cleaned data, finds neighbor pairs within ``max_hamming`` mismatches
    in sequence space and interpolates between each pair to produce soft
    one-hot samples:
        feature = lambda * onehot_i + (1 - lambda) * onehot_j
        label   = lambda * y_i     + (1 - lambda) * y_j

    Args:
        features: one-hot encoded tensor, shape (N, seq_len, 4)
        labels: label tensor, shape (N,)
        seqs: list of string sequences (used to find neighbor pairs)
        lambdas: interpolation positions, default [0.5]
        max_hamming: maximum Hamming distance, default 1

    Returns:
        (all_features, all_labels, aug_seqs, n_pairs)
    """
    if lambdas is None:
        lambdas = [0.5]

    N, L, C = features.shape
    BASES = "ATGC"

    # Build a sequence -> index mapping
    seq_to_idx = {}
    for i, s in enumerate(seqs):
        s_upper = s.upper()
        if s_upper not in seq_to_idx:
            seq_to_idx[s_upper] = []
        seq_to_idx[s_upper].append(i)

    # Generate variant sequences with Hamming distance <= max_d
    def _hamming_variants(s, max_d):
        """Return the set of variant sequences within Hamming distance <= max_d."""
        results = set()
        # Recursive generation: collect all possible position combinations
        from itertools import combinations
        for d in range(1, max_d + 1):
            for positions in combinations(range(len(s)), d):
                # Enumerate all base substitutions at each position combination
                _gen_variants(s, list(positions), 0, results)
        return results

    def _gen_variants(s, positions, idx, results):
        if idx >= len(positions):
            results.add(s)
            return
        pos = positions[idx]
        orig = s[pos]
        for b in BASES:
            if b != orig:
                _gen_variants(s[:pos] + b + s[pos + 1:], positions, idx + 1, results)

    # Find all neighbor pairs
    pairs = []
    seen = set()
    for i, s in enumerate(seqs):
        s_upper = s.upper()
        variants = _hamming_variants(s_upper, max_hamming)
        for nb in variants:
            if nb in seq_to_idx:
                for j in seq_to_idx[nb]:
                    if j > i and (i, j) not in seen:
                        seen.add((i, j))
                        pairs.append((i, j))

    n_pairs = len(pairs)
    n_interp = n_pairs * len(lambdas)
    print(f"  [NeighborInterp] {n_pairs} Hamming<={max_hamming} neighbor pairs, "
          f"{len(lambdas)} interpolation points -> +{n_interp} virtual samples")

    # Generate interpolation samples
    aug_features = torch.zeros(n_interp, L, C, dtype=features.dtype)
    aug_labels = torch.zeros(n_interp, dtype=labels.dtype)

    idx = 0
    for lam in lambdas:
        for i, j in pairs:
            aug_features[idx] = lam * features[i] + (1 - lam) * features[j]
            aug_labels[idx] = lam * labels[i] + (1 - lam) * labels[j]
            idx += 1

    # Concatenate
    all_features = torch.cat([features, aug_features], dim=0)
    all_labels = torch.cat([labels, aug_labels], dim=0)
    # Extend the seqs list (interpolated samples have no real sequence; use a placeholder)
    aug_seqs = ["INTERP"] * n_interp

    return all_features, all_labels, aug_seqs, n_pairs
