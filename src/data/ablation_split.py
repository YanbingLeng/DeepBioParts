"""Data splitting utilities for ablation experiments.

Provides three capabilities used by ``scripts/train_predictor.py`` in ablation mode:

1. Load the **fixed external test set** from ``results/test_set_results/`` (read-only; never recomputed or overwritten).
2. Exclude sequences overlapping the fixed test set from the training pool; raise an error if any overlap remains.
3. Get or create the cluster-based five-fold split manifest (computed and persisted to disk on first use, then reused directly by subsequent variants).

All variants share the same fixed test set and the same five-fold split, ensuring that ablation results are comparable.

Constraints:
- Regression tasks read only ``sequence``/``activity``; classification tasks read only ``sequence``/``label``.
- ``pred_*`` / ``prob_*`` columns contain predictions from existing models and must never be used as training inputs.
- The original row order of the test CSV is preserved; ``sample_id`` = original row index, used for pairing across variants.
- The test set is excluded from training/early stopping/model selection/hyperparameter tuning.
- Five-fold cluster-based CV is performed only within the training pool after the fixed test set has been excluded.
"""

import os
import zlib
import numpy as np
import pandas as pd

from src.utils.data import ClusterBasedKFold


# Fixed test set filename for each (biopart, task_type) pair (under fixed_test_dir)
FIXED_TEST_FILES = {
    ('promoter', 'regression'): 'test_promoter.csv',
    ('rbs', 'regression'): 'test_rbs.csv',
    ('terminator', 'regression'): 'test_terminator_reg.csv',
    ('terminator', 'classification'): 'test_terminator.csv',
}

# Similarity fields passed through unchanged to the output; present in regression test sets only
REGRESSION_PASSTHROUGH_COLS = ('max_3mer_cosine_sim', 'sim_bin', 'min_hamming_to_train')


def _standardize(seq: str) -> str:
    """Standardize a sequence: strip whitespace and uppercase. Used for cross-set deduplication and overlap detection."""
    return str(seq).strip().upper()


def _stable_sample_id(seq: str) -> int:
    """Stable sample_id for training sequences (crc32; consistent across runs and variants)."""
    return zlib.crc32(_standardize(seq).encode('utf-8'))


def get_fixed_test_filename(biopart: str, task_type: str) -> str:
    """Return the fixed test set filename."""
    key = (biopart, task_type)
    if key not in FIXED_TEST_FILES:
        raise ValueError(
            f"No fixed test set registered for biopart={biopart!r}, task_type={task_type!r}. "
            f"Known: {list(FIXED_TEST_FILES.keys())}"
        )
    return FIXED_TEST_FILES[key]


def load_fixed_test_set(biopart, task_type, fixed_test_dir='results/test_set_results'):
    """Load the fixed external test set (read-only).

    Args:
        biopart: 'promoter' / 'rbs' / 'terminator'
        task_type: 'regression' / 'classification'
        fixed_test_dir: directory containing the fixed test sets

    Returns:
        seqs: list[str], in the original row order
        labels: np.ndarray (float32 for regression, int64 for classification)
        meta_df: pandas.DataFrame aligned with seqs, containing:
            - ``sample_id``: original row index (0-based), used for pairing across variants
            - similarity passthrough columns present in regression tasks
              (e.g. ``max_3mer_cosine_sim``); empty for classification
        raw_df: DataFrame as read (original row order preserved, only the required columns)

    Notes:
        Reads only ``sequence`` + (``activity``|``label``); ``pred_*``/``prob_*`` columns are never read.
    """
    fname = get_fixed_test_filename(biopart, task_type)
    path = os.path.join(fixed_test_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fixed test set not found: {path}. Ablation experiments must use the existing test set in results/test_set_results/."
        )
    raw_df = pd.read_csv(path)
    if 'sequence' not in raw_df.columns:
        raise ValueError(f"Fixed test set {path} is missing the 'sequence' column")

    seqs = [_standardize(s) for s in raw_df['sequence'].astype(str).tolist()]

    if task_type == 'classification':
        if 'label' not in raw_df.columns:
            raise ValueError(f"Classification test set {path} is missing the 'label' column")
        labels = raw_df['label'].astype(np.int64).values
        label_col = 'label'
    else:
        if 'activity' not in raw_df.columns:
            raise ValueError(f"Regression test set {path} is missing the 'activity' column")
        labels = raw_df['activity'].astype(np.float32).values
        label_col = 'activity'

    # Passthrough columns: similarity fields present only in regression test sets;
    # classification test sets lack them, so meta contains only sample_id
    meta_cols = ['sample_id']
    passthrough = {}
    for c in REGRESSION_PASSTHROUGH_COLS:
        if c in raw_df.columns:
            passthrough[c] = raw_df[c].values
            meta_cols.append(c)

    meta_df = pd.DataFrame({'sample_id': np.arange(len(seqs)), **passthrough})

    # Keep only the columns that must be read (avoid carrying pred_*/prob_* downstream)
    slim_df = raw_df[['sequence', label_col]].copy()
    slim_df['sequence'] = seqs
    slim_df[label_col] = labels

    return seqs, labels, meta_df, slim_df


def exclude_test_from_train(train_seqs, train_labels, test_seqs):
    """Exclude every sequence present in the fixed test set from the training pool.

    Args:
        train_seqs: list of training sequences
        train_labels: training labels (list/np.ndarray)
        test_seqs: fixed test set sequences

    Returns:
        kept_seqs, kept_labels: training pool after exclusion

    Raises:
        RuntimeError: if the training pool still overlaps the test set after exclusion
            (should not happen in theory; if it does, the upstream data contains
            duplicates — abort immediately and report the overlap count).
    """
    test_set = {_standardize(s) for s in test_seqs}

    train_labels = np.asarray(train_labels)
    kept_seqs, kept_idx = [], []
    for i, s in enumerate(train_seqs):
        if _standardize(s) not in test_set:
            kept_seqs.append(s)
            kept_idx.append(i)
    kept_labels = train_labels[np.array(kept_idx, dtype=np.int64)] if kept_idx else train_labels[:0]

    # Defensive check: no overlap should remain after exclusion
    kept_set = {_standardize(s) for s in kept_seqs}
    overlap = test_set & kept_set
    if overlap:
        raise RuntimeError(
            f"Training pool still overlaps the fixed test set: {len(overlap)} sequences. Aborted to prevent data leakage."
        )

    removed = len(train_seqs) - len(kept_seqs)
    print(f"[FixedTest] Training pool {len(train_seqs)} -> {len(kept_seqs)} (removed {removed} sequences overlapping the test set)")
    return kept_seqs, list(kept_labels)


def _manifest_filename(biopart, task_type, similarity_threshold, kmer_size, n_folds):
    """Manifest filename uniquely keyed by (biopart, task, similarity threshold, k-mer size, n_folds)."""
    return (
        f"{biopart}_{task_type}"
        f"_t{similarity_threshold:g}_k{kmer_size}_f{n_folds}_manifest.csv"
    )


def _build_fold_assignment(seqs, similarity_threshold, kmer_size, n_folds):
    """Run cluster-based K-fold on the training pool; return each sample's validation fold (1..n_folds)."""
    kf = ClusterBasedKFold(
        n_splits=n_folds,
        similarity_threshold=similarity_threshold,
        kmer_size=kmer_size,
        random_state=42,
        verbose=True,
    )
    kf._fit(seqs)
    assignment = np.zeros(len(seqs), dtype=np.int64)
    for k, (train_idx, val_idx) in enumerate(kf.split(seqs), start=1):
        assignment[val_idx] = k
    return assignment


def get_or_create_manifest(
    biopart, task_type, train_pool_seqs, train_pool_labels,
    test_seqs, similarity_threshold=0.8, kmer_size=3, n_folds=5,
    manifest_dir='results/test_set_results/ablation_manifests',
):
    """Get or create the cluster-based five-fold split manifest.

    First call: cluster the training pool **after excluding the fixed test set**,
    generate the manifest, and persist it to disk.
    Subsequent variants: read the existing manifest directly without re-clustering
    (guarantees an identical split).

    manifest CSV columns: ``sample_id, sequence, outer_split, fold``
      - Training pool rows: ``outer_split='trainval'``, ``fold`` ∈ {1..n_folds}
        (the fold in which the sample serves as validation)
      - Test set rows: ``outer_split='test'``, ``fold`` is empty
        (the test set does not participate in CV)

    Returns:
        fold_assignment: np.ndarray[int] of length len(train_pool_seqs), each element in 1..n_folds
    """
    os.makedirs(manifest_dir, exist_ok=True)
    fname = _manifest_filename(biopart, task_type, similarity_threshold, kmer_size, n_folds)
    manifest_path = os.path.join(manifest_dir, fname)

    train_pool_seqs = [_standardize(s) for s in train_pool_seqs]

    if os.path.exists(manifest_path):
        print(f"[Manifest] Reusing existing split: {manifest_path}")
        df = pd.read_csv(manifest_path)
        trainval_df = df[df['outer_split'] == 'trainval'].copy()
        # Align via sample_id (crc32) to ensure the training pool matches the manifest
        sid_to_fold = dict(zip(trainval_df['sample_id'].astype(np.int64), trainval_df['fold'].astype(np.int64)))
        fold_assignment = np.array(
            [sid_to_fold[_stable_sample_id(s)] for s in train_pool_seqs],
            dtype=np.int64,
        )
        if np.any(fold_assignment < 1) or np.any(fold_assignment > n_folds):
            raise RuntimeError(
                "Manifest is inconsistent with the current training pool: some samples have no fold assignment. "
                "Delete the manifest to regenerate it, or verify that the training data has not changed."
            )
        return fold_assignment

    # First use: cluster to generate the split
    print(f"[Manifest] Generating cluster-based split for the first time: {manifest_path}")
    fold_assignment = _build_fold_assignment(train_pool_seqs, similarity_threshold, kmer_size, n_folds)

    rows = []
    for seq, fold in zip(train_pool_seqs, fold_assignment):
        rows.append({
            'sample_id': _stable_sample_id(seq),
            'sequence': seq,
            'outer_split': 'trainval',
            'fold': int(fold),
        })
    # Test set rows (fold left empty)
    for i, seq in enumerate(test_seqs):
        rows.append({
            'sample_id': i,  # test set sample_id = original row index
            'sequence': _standardize(seq),
            'outer_split': 'test',
            'fold': '',
        })
    out_df = pd.DataFrame(rows, columns=['sample_id', 'sequence', 'outer_split', 'fold'])
    out_df.to_csv(manifest_path, index=False)
    print(f"[Manifest] Saved: {manifest_path} (trainval={len(train_pool_seqs)}, test={len(test_seqs)})")

    return fold_assignment
