"""
.. note::
    The predictor training logic (``Predictor_language``) is defined in this module.
    Supports construction from config: ``Predictor_language.from_config(cfg)``.
"""

import random
import os
import sys
import json
import time
import functools
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import torch
from torch import nn
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from utils.utils import EarlyStopping_P
from utils.data import seq2onehot, onehot2seq
from utils.module import ORNet, BidirectionalLSTM, TransitionLayer, DenseBlock, BottleneckLayer
import matplotlib.pyplot as plt
from models.predictor import ModelFactory
from visualization.training_plots import plot_fold_metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
from collections import defaultdict
import warnings
from metrics import (
    compute_regression_metrics as calculate_regression_metrics,
    compute_classification_metrics as calculate_classification_metrics,
    weighted_spearman,
    ordinal_accuracy as ordinal_classification_accuracy,
)


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*"
)

def regression_loss(pred, target, l1_weight=0.1):
    """Regression loss: MSE + L1 regularization."""
    mse_loss = F.mse_loss(pred, target)
    l1_loss = F.l1_loss(pred, target)
    return mse_loss + l1_weight * l1_loss

def ordinal_regression_loss(logits, labels, biases, sample_weights=None):
    """
    Ordinal regression loss (consistent with predictor.py).

    Args:
        logits: [batch_size, K] where K = num_classes - 1
        labels: [batch_size] class labels (0, 1, ..., num_classes-1)
        biases: [K] or_bias parameters
        sample_weights: [batch_size] optional per-sample weights

    Returns:
        total_loss: the total loss
    """
    K = logits.size(1)  # nclass-1
    batch_size = labels.size(0)

    indices = torch.arange(K, device=labels.device).unsqueeze(0)  # [1, K]
    targets = (labels.unsqueeze(1) > indices).float()  # [batch_size, K]

    # binary_cross_entropy_with_logits is more numerically stable than sigmoid + BCE
    bce_per_element = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [batch_size, K]

    if sample_weights is not None:
        bce_per_element = bce_per_element * sample_weights.unsqueeze(1)

    out_loss = bce_per_element.sum(dim=1).mean()

    # Auxiliary loss to ensure or_bias is in descending order and differences are controlled
    aux_loss = 0.0
    desired_diff = 5.0  # desired per-step gap
    if len(biases) > 1:
        for i in range(len(biases) - 1):
            # Penalize if the difference deviates from desired_diff
            aux_loss += F.softplus(desired_diff - (biases[i] - biases[i+1]))

    total_loss = out_loss + aux_loss
    return total_loss

def classification_loss(pred, target):
    """Classification loss: cross-entropy (deprecated, kept for backward compatibility)."""
    # pred: [batch_size, num_classes]
    # target: [batch_size] (class index, 0 or 1)
    return F.cross_entropy(pred, target.long())

def logits2classes(logits):
    """
    Convert ordinal-regression logits to class predictions.

    Args:
        logits: [batch_size, num_classes-1] logits

    Returns:
        classes: [batch_size] predicted class indices
    """
    probs = torch.sigmoid(logits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.cat((torch.ones((probs.shape[0],1)).to(device),
                   probs,
                   torch.zeros((probs.shape[0],1)).to(device)),
                  dim=1)
    x = x[:,:-1] - x[:,1:]
    classes = torch.argmax(x, dim=1)

    return classes

def calculate_r2_by_bins(y_true, y_pred, num_bins=5, min_val=0.0, max_val=1.0):
    """
    Compute R^2 stratified by label bins.

    Args:
        y_true: ground-truth label array
        y_pred: predicted label array
        num_bins: number of bins
        min_val: minimum label value
        max_val: maximum label value

    Returns:
        bins_info: list of dicts, one per bin
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Define bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    bins_info = []

    for i in range(num_bins):
        bin_min = bin_edges[i]
        bin_max = bin_edges[i + 1]

        # Sample indices in this bin; last bin includes the right edge
        if i == num_bins - 1:
            mask = (y_true >= bin_min) & (y_true <= bin_max)
        else:
            mask = (y_true >= bin_min) & (y_true < bin_max)

        bin_true = y_true[mask]
        bin_pred = y_pred[mask]

        if len(bin_true) > 1 and np.var(bin_true) > 0:
            # R^2
            ss_res = np.sum((bin_true - bin_pred) ** 2)
            ss_tot = np.sum((bin_true - np.mean(bin_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Other metrics
            mse = np.mean((bin_true - bin_pred) ** 2)
            mae = np.mean(np.abs(bin_true - bin_pred))

            bins_info.append({
                'bin_index': i + 1,
                'bin_range': f'[{bin_min:.3f}, {bin_max:.3f})' if i < num_bins - 1 else f'[{bin_min:.3f}, {bin_max:.3f}]',
                'sample_count': len(bin_true),
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'mean_true': np.mean(bin_true),
                'mean_pred': np.mean(bin_pred),
                'std_true': np.std(bin_true),
                'std_pred': np.std(bin_pred)
            })
        else:
            # Insufficient samples or zero variance
            bins_info.append({
                'bin_index': i + 1,
                'bin_range': f'[{bin_min:.3f}, {bin_max:.3f})' if i < num_bins - 1 else f'[{bin_min:.3f}, {bin_max:.3f}]',
                'sample_count': len(bin_true),
                'r2': np.nan,
                'mse': np.nan,
                'mae': np.nan,
                'mean_true': np.mean(bin_true) if len(bin_true) > 0 else np.nan,
                'mean_pred': np.mean(bin_pred) if len(bin_pred) > 0 else np.nan,
                'std_true': np.nan,
                'std_pred': np.nan
            })

    return bins_info

class SequenceData(Dataset):
    def __init__(self, data, label):
        self.data = data  # list of (variable-length) sequences
        self.target = label

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

    def __getdata__(self):
        return self.data, self.target

class TestData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __getdata__(self):
        return self.data

class Predictor_language:
    def __init__(self,
                 seq_len=None,
                 num_classes=2,  # number of classes for classification
                 batch_size=64,
                 model_name = None,
                 model_type = None,
                 epoch=200,
                 patience=50,
                 log_steps=10,
                 save_steps=20,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 clip_grad_norm=1.0,
                 dropout_rate=0.2,
                 conv_width_motif=5,
                 n_heads=16,
                 conv_hidden=128,
                 motif_conv_hidden=256,
                 optimizer='adam',
                 vocab_size=None,
                 encoding_type='onehot',
                 ensemble_method='mean',
                 l1_loss_weight=0.1,
                 task_type='regression',
                 fold_model_paths = None,
                 checkpoint_root = None,
                 use_cluster_split=True,  # cluster-based split to prevent sequence-similarity leakage
                 similarity_threshold=0.8,  # 3-mer cosine similarity threshold for clustering
                 kmer_size=3,  # k-mer size
                 n_folds=5,        # number of cross-validation folds
                 use_log_label=False,  # apply log10(label+1) transform (regression only)
                 use_mixup=False,      # enable Mixup augmentation
                 mixup_alpha=0.2,      # Mixup Beta distribution parameter
                 use_mutation_augment=False,  # enable random-mutation augmentation
                 mutation_n=1,         # number of mutated positions per sequence
                 mutation_copies=1,    # number of augmented copies (1 = 2x, 2 = 3x)
                 use_neighbor_interp=False,  # enable neighbor linear interpolation
                 interp_lambdas=None,  # interpolation positions, default [0.5]
                 interp_max_hamming=1,  # max Hamming distance for interpolation neighbors
                 ablation_variant='full',  # CNN–Attention–BiLSTM ablation variant (effective for model_type=conv only)
                 seed=42,  # random seed (passed to train() to control the RNG)
                 ):

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.encoding_type = encoding_type

        # Store model configuration parameters
        self.model_type = model_type
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.conv_width_motif = conv_width_motif
        self.n_heads = n_heads
        self.conv_hidden = conv_hidden
        self.motif_conv_hidden = motif_conv_hidden
        self.fold_model_paths = fold_model_paths
        # Do not instantiate the model at init time (saves memory)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        self.ensemble_method = ensemble_method
        self.l1_loss_weight = l1_loss_weight
        self.task_type = task_type  # 'regression' or 'classification'
        self.checkpoint_root = checkpoint_root  # checkpoint root directory
        self.use_cluster_split = use_cluster_split  # use cluster-based split
        self.similarity_threshold = similarity_threshold  # similarity threshold
        self.kmer_size = kmer_size  # k-mer size
        self.n_folds = n_folds  # number of cross-validation folds
        self.use_log_label = use_log_label  # log10 label transform
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_mutation_augment = use_mutation_augment
        self.mutation_n = mutation_n
        self.mutation_copies = mutation_copies
        self.use_neighbor_interp = use_neighbor_interp
        self.interp_lambdas = interp_lambdas or [0.5]
        self.interp_max_hamming = interp_max_hamming
        self.ablation_variant = ablation_variant  # CNN–Attention–BiLSTM ablation variant
        self.seed = seed  # random seed (used to seed the RNG inside train())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_config(cls, cfg: dict, **overrides):
        """Create a Predictor_language from a layered config dict.

        Args:
            cfg: Configuration dictionary (e.g. from ``core.config.load_config``).
            **overrides: Keyword overrides with highest priority.

        Returns:
            Predictor_language instance with parameters from config.
        """
        model_cfg = cfg.get('model', {})
        train_cfg = cfg.get('training', {})
        seq_cfg = cfg.get('sequence', {})
        data_cfg = cfg.get('data', {})

        kwargs = dict(
            seq_len=seq_cfg.get('seq_len'),
            num_classes=train_cfg.get('num_classes', 2),
            batch_size=train_cfg.get('batch_size', 64),
            model_type=model_cfg.get('architecture'),
            epoch=train_cfg.get('epochs', 200),
            patience=train_cfg.get('patience', 100),
            log_steps=train_cfg.get('log_steps', 10),
            save_steps=train_cfg.get('save_steps', 20),
            learning_rate=train_cfg.get('learning_rate', 1e-4),
            weight_decay=train_cfg.get('weight_decay', 1e-4),
            clip_grad_norm=train_cfg.get('clip_grad_norm', 1.0),
            dropout_rate=model_cfg.get('dropout_rate', 0.2),
            conv_width_motif=model_cfg.get('conv_width_motif', 5),
            n_heads=model_cfg.get('n_heads', 16),
            conv_hidden=model_cfg.get('conv_hidden', 128),
            motif_conv_hidden=model_cfg.get('motif_conv_hidden', 256),
            optimizer=train_cfg.get('optimizer', 'adam'),
            encoding_type=seq_cfg.get('encoding', 'onehot'),
            ensemble_method=train_cfg.get('ensemble_method', 'mean'),
            l1_loss_weight=train_cfg.get('l1_weight', 0.1),
            task_type=train_cfg.get('task_type', 'regression'),
            use_cluster_split=data_cfg.get('use_cluster_split', False),
            similarity_threshold=data_cfg.get('similarity_threshold', 0.8),
            kmer_size=data_cfg.get('kmer_size', 3),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def plot_metrics_curves(self, fold_metrics, save_dir):
        """Plot and save per-fold validation metric curves."""
        # Convert {fold_num: [{metric: value}, ...]} to [{metric: [values]}, ...]
        fold_histories = []
        for _fnum, metrics_list in fold_metrics.items():
            history = {}
            if metrics_list:
                for key in metrics_list[0]:
                    history[key] = [m.get(key, float('nan')) for m in metrics_list]
            fold_histories.append(history)
        plot_fold_metrics(fold_histories, save_dir, task_type=self.task_type)

    def train(self, seqs, labels, savepath, trial=None, fixed_test=None, fold_assignment=None,
              single_split=False):
        """
        Train the model with 5-fold cross-validation on the training set only.
        The test set is NOT evaluated inside this method.

        Args:
            fixed_test: ablation mode only; a (test_seqs, test_labels) tuple. When provided,
                the fixed external test set is used and the internal cluster split is skipped.
                None falls back to the standard logic.
            fold_assignment: ablation mode only; an array of length len(seqs) with values in
                1..n_folds (the fold in which each sample serves as validation). When provided,
                internal KFold clustering is skipped.
            single_split: in ablation mode, skip K-fold CV and train a single model with
                fold==1 as validation and the rest as training (5x speed-up).
                Requires fold_assignment.

        Returns:
            results: cross-validation results dict
            test_feature: test-set features
            test_label: test-set labels
        """
        # Set random seeds (use the instance seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.seqs = seqs
        self.labels = np.array(labels, dtype=np.float32)
        self.checkpoint_root = savepath
        # Keep raw sequences (used later for similarity-stratified reporting)
        self._all_seqs_raw = list(seqs)

        # Ablation mode: disable sample-count-changing augmentation before it runs,
        # ensuring strict alignment with fold_assignment
        if fixed_test is not None and (self.use_mutation_augment or self.use_neighbor_interp):
            print("[Ablation] Fixed-test-set mode: disabling mutation/neighbor_interp augmentation (to stay aligned with the manifest)")
            self.use_mutation_augment = False
            self.use_neighbor_interp = False

        # log10 label transform (regression only)
        if self.use_log_label and self.task_type == 'regression':
            self.labels = np.log10(self.labels + 1.0).astype(np.float32)
            os.makedirs(savepath, exist_ok=True)
            with open(os.path.join(savepath, 'label_transform.json'), 'w') as f:
                json.dump({'transform': 'log10', 'shift': 1.0}, f, indent=2)
            print(f"[Label Transform] log10(label+1), range: [{self.labels.min():.4f}, {self.labels.max():.4f}]")
        else:
            self.use_log_label = False

        filename_sim = savepath

        if not os.path.exists(filename_sim):
            os.makedirs(filename_sim)

        # Encode sequences according to encoding_type
        if self.encoding_type == 'onehot':
            total_feature = seq2onehot(self.seqs)
            total_feature = torch.tensor(total_feature, dtype=torch.float32)
        else:
            if not hasattr(self, '_encoded_data'):
                from data.sequence_encoding import encode_sequences
                encoded_seqs, _, _, _ = encode_sequences(self.seqs, self.encoding_type)
                max_len = max(len(seq) for seq in encoded_seqs)
                padded_seqs = []
                for seq in encoded_seqs:
                    if len(seq) < max_len:
                        padding_token = 0
                        padding = np.full(max_len - len(seq), padding_token)
                        seq = np.concatenate([seq, padding])
                    padded_seqs.append(seq)
                self._encoded_data = torch.tensor(np.array(padded_seqs), dtype=torch.long)
            total_feature = self._encoded_data

        # [Data-leakage fix #1] Split the data BEFORE any augmentation/normalization
        # Convert raw labels to a tensor
        total_label = torch.tensor(self.labels, dtype=torch.float32)

        # Random-mutation augmentation (one-hot only)
        if self.use_mutation_augment and self.encoding_type == 'onehot':
            from data.augmentation import random_mutate_onehot
            orig_n = len(total_feature)
            all_aug_features = [total_feature]
            all_aug_labels = [total_label]
            all_aug_seqs = [list(self.seqs)]
            for _ in range(self.mutation_copies):
                aug_f = random_mutate_onehot(total_feature, n_mutations=self.mutation_n)
                all_aug_features.append(aug_f)
                all_aug_labels.append(total_label.clone())
                all_aug_seqs.append(list(self.seqs))
            total_feature = torch.cat(all_aug_features, dim=0)
            total_label = torch.cat(all_aug_labels, dim=0)
            self.seqs = sum(all_aug_seqs, [])
            print(f"[Mutation Augment] {self.mutation_n} positions x {self.mutation_copies} copies, "
                  f"sample count {orig_n} -> {len(total_feature)}")

        # Neighbor linear interpolation (one-hot only)
        if self.use_neighbor_interp and self.encoding_type == 'onehot':
            from data.augmentation import neighbor_interpolate
            orig_n = len(total_feature)
            total_feature, total_label, aug_seqs, n_pairs = neighbor_interpolate(
                total_feature, total_label, list(self.seqs),
                lambdas=self.interp_lambdas,
                max_hamming=self.interp_max_hamming,
            )
            self.seqs = list(self.seqs) + aug_seqs
            print(f"[Neighbor Interp] {n_pairs} neighbor pairs x {len(self.interp_lambdas)} interpolation points, "
                  f"sample count {orig_n} -> {len(total_feature)}")

        # Keep self.labels in sync with the augmented sample count (cluster_based_split depends on it)
        if self.use_mutation_augment or self.use_neighbor_interp:
            self.labels = total_label.numpy().astype(np.float32)

        # ===== Ablation mode: fixed external test set + precomputed fold assignment =====
        _ablation_mode = fixed_test is not None
        _precomputed_splits = None
        if _ablation_mode:
            # Disable sample-count-changing augmentation so fold_assignment stays
            # strictly aligned with the training pool
            if self.use_mutation_augment or self.use_neighbor_interp:
                print("[Ablation] Fixed-test-set mode: disabling mutation/neighbor_interp augmentation (to stay aligned with the manifest)")
                self.use_mutation_augment = False
                self.use_neighbor_interp = False
            # Training pool = all passed-in sequences (the caller has already excluded
            # samples overlapping the fixed test set)
            train_feature = total_feature
            test_seqs_fixed = list(fixed_test[0])
            if self.task_type == 'classification':
                train_label = total_label.numpy().astype(np.int64)
                test_label = np.array(fixed_test[1], dtype=np.int64)
            else:
                train_label = total_label.numpy().astype(np.float32)
                test_label = np.array(fixed_test[1], dtype=np.float32)
            # Encode the fixed test set (preserving the original row order)
            if self.encoding_type == 'onehot':
                test_feature = torch.tensor(seq2onehot(test_seqs_fixed), dtype=torch.float32)
            else:
                from data.sequence_encoding import encode_sequences
                enc, _, _, _ = encode_sequences(test_seqs_fixed, self.encoding_type)
                max_len = max(max(len(s) for s in enc), max(len(s) for s in encode_sequences(self.seqs, self.encoding_type)[0]))
                padded = []
                for s in enc:
                    if len(s) < max_len:
                        s = np.concatenate([s, np.full(max_len - len(s), 0)])
                    padded.append(s)
                test_feature = torch.tensor(np.array(padded), dtype=torch.long)
            self._original_train_seqs = list(self.seqs)
            self._train_seqs_for_similarity = list(self.seqs)
            # Build (train_idx, val_idx) from fold_assignment
            fa = np.asarray(fold_assignment)
            if len(fa) != len(train_feature):
                raise ValueError(
                    f"fold_assignment length ({len(fa)}) does not match the training pool size ({len(train_feature)})"
                )
            if single_split:
                # No K-fold CV: use fold==1 as validation and the rest as training, training a single model
                _precomputed_splits = [(np.where(fa != 1)[0], np.where(fa == 1)[0])]
                print(f"[Ablation] single-split mode: val=fold1(n={(_precomputed_splits[0][1]).size}), "
                      f"train=rest(n={(_precomputed_splits[0][0]).size})")
            else:
                _precomputed_splits = [
                    (np.where(fa != k)[0], np.where(fa == k)[0]) for k in range(1, self.n_folds + 1)
                ]
            print(f"[Ablation] Fixed test set size={len(test_seqs_fixed)}, training pool={len(train_feature)}, "
                  f"will train {len(_precomputed_splits)} model(s)")

        # Process labels according to task type
        if _ablation_mode:
            pass  # fixed-test-set mode: train_feature/test_feature/labels already prepared above
        elif self.task_type == 'classification':
            # Classification: split first, then cast to integer type
            if self.use_cluster_split:
                # Sequence-similarity cluster-based split
                from utils.data import cluster_based_split

                print("\n" + "="*70)
                print("Splitting dataset by sequence-similarity clustering (classification mode)")
                print("="*70)

                # Raw sequences
                original_seqs = self.seqs

                # Split by sequence, then re-encode features
                train_seqs, train_labels_list, test_seqs, test_labels_list = cluster_based_split(
                    seqs=original_seqs,
                    labels=self.labels.tolist(),
                    test_size=0.2,
                    similarity_threshold=self.similarity_threshold,
                    kmer_size=self.kmer_size,
                    random_state=42,
                    verbose=True
                )

                # Re-encode features from the split sequences
                if self.encoding_type == 'onehot':
                    train_feature = seq2onehot(train_seqs)
                    train_feature = torch.tensor(train_feature, dtype=torch.float32)
                    test_feature = seq2onehot(test_seqs)
                    test_feature = torch.tensor(test_feature, dtype=torch.float32)
                else:
                    # Tokenized encoding
                    from data.sequence_encoding import encode_sequences
                    train_encoded, _, _, _ = encode_sequences(train_seqs, self.encoding_type)
                    test_encoded, _, _, _ = encode_sequences(test_seqs, self.encoding_type)

                    # Pad to the same length
                    max_len = max(max(len(seq) for seq in train_encoded),
                                 max(len(seq) for seq in test_encoded))
                    padded_train = []
                    padded_test = []
                    for seq in train_encoded:
                        if len(seq) < max_len:
                            padding = np.full(max_len - len(seq), 0)
                            seq = np.concatenate([seq, padding])
                        padded_train.append(seq)
                    for seq in test_encoded:
                        if len(seq) < max_len:
                            padding = np.full(max_len - len(seq), 0)
                            seq = np.concatenate([seq, padding])
                        padded_test.append(seq)

                    train_feature = torch.tensor(np.array(padded_train), dtype=torch.long)
                    test_feature = torch.tensor(np.array(padded_test), dtype=torch.long)

                # Cast labels to integer type
                train_label = np.array(train_labels_list, dtype=np.int64)
                test_label = np.array(test_labels_list, dtype=np.int64)

                print(f"\nCluster-based split complete:")
                print(f"  Training set size: {len(train_seqs)}")
                print(f"  Test set size: {len(test_seqs)}")
                print("="*70 + "\n")

                # Save raw training sequences for K-fold CV
                self._original_train_seqs = train_seqs
            else:
                # Random split (legacy)
                train_feature, test_feature, train_label, test_label = train_test_split(
                    total_feature, total_label, test_size=0.2, random_state=42
                )
                # Cast to integer type
                train_label = train_label.numpy().astype(np.int64)
                test_label = test_label.numpy().astype(np.int64)

            print(f"Classification mode: training label range [{np.min(train_label)}, {np.max(train_label)}], "
                  f"class distribution: {np.bincount(train_label)}")

        else:  # Regression
            # [Data-leakage fix #1] Regression: cluster-split first, then normalize
            if self.use_cluster_split:
                # Sequence-similarity cluster-based split
                from utils.data import cluster_based_split

                print("\n" + "="*70)
                print("Splitting dataset by sequence-similarity clustering (regression mode)")
                print("="*70)

                # Raw sequences
                original_seqs = self.seqs

                # Split by sequence, then re-encode features
                train_seqs, train_labels_list, test_seqs, test_labels_list = cluster_based_split(
                    seqs=original_seqs,
                    labels=self.labels.tolist(),
                    test_size=0.2,
                    similarity_threshold=self.similarity_threshold,
                    kmer_size=self.kmer_size,
                    random_state=42,
                    verbose=True
                )

                # Re-encode features from the split sequences
                if self.encoding_type == 'onehot':
                    train_feature = seq2onehot(train_seqs)
                    train_feature = torch.tensor(train_feature, dtype=torch.float32)
                    test_feature = seq2onehot(test_seqs)
                    test_feature = torch.tensor(test_feature, dtype=torch.float32)
                else:
                    # Tokenized encoding
                    from data.sequence_encoding import encode_sequences
                    train_encoded, _, _, _ = encode_sequences(train_seqs, self.encoding_type)
                    test_encoded, _, _, _ = encode_sequences(test_seqs, self.encoding_type)

                    # Pad to the same length
                    max_len = max(max(len(seq) for seq in train_encoded),
                                 max(len(seq) for seq in test_encoded))
                    padded_train = []
                    padded_test = []
                    for seq in train_encoded:
                        if len(seq) < max_len:
                            padding = np.full(max_len - len(seq), 0)
                            seq = np.concatenate([seq, padding])
                        padded_train.append(seq)
                    for seq in test_encoded:
                        if len(seq) < max_len:
                            padding = np.full(max_len - len(seq), 0)
                            seq = np.concatenate([seq, padding])
                        padded_test.append(seq)

                    train_feature = torch.tensor(np.array(padded_train), dtype=torch.long)
                    test_feature = torch.tensor(np.array(padded_test), dtype=torch.long)

                # Cast labels
                train_label = np.array(train_labels_list, dtype=np.float32)
                test_label = np.array(test_labels_list, dtype=np.float32)
                train_label_np = train_label
                test_label_np = test_label

                print(f"\nCluster-based split complete:")
                print(f"  Training set size: {len(train_seqs)}")
                print(f"  Test set size: {len(test_seqs)}")
                print("="*70 + "\n")

                # Save raw training sequences for K-fold CV
                self._original_train_seqs = train_seqs
            else:
                # Random split (legacy)
                train_feature, test_feature, train_label, test_label = train_test_split(
                    total_feature, total_label, test_size=0.2, random_state=42
                )
                train_label_np = train_label.numpy()
                test_label_np = test_label.numpy()

            # Regression: use raw label values directly
            print(f"Regression mode - using raw label values:")
            print(f"  Training label range: [{np.min(train_label_np):.4f}, {np.max(train_label_np):.4f}]")
            print(f"  Test label range: [{np.min(test_label_np):.4f}, {np.max(test_label_np):.4f}]")

        # Keep test data for later use
        self.test_feature = test_feature
        self.test_label = test_label

        # Keep training sequences for similarity-stratified reporting
        if hasattr(self, '_original_train_seqs'):
            self._train_seqs_for_similarity = self._original_train_seqs
        else:
            # Recover training sequences from one-hot encoding
            _train_seqs_for_sim = onehot2seq(train_feature)
            self._train_seqs_for_similarity = _train_seqs_for_sim

        # Choose K-fold splitting strategy based on use_cluster_split
        if _ablation_mode:
            _use_single_split = False  # _precomputed_splits is ready; skip kf
        elif self.n_folds <= 1:
            # Single-split mode: hold out 15% of the training set as validation
            print("\nSingle-split mode: holding out 15% of training set as validation")
            train_feature_cv, val_feature, train_label_cv, val_label = train_test_split(
                train_feature, train_label, test_size=0.15, random_state=42
            )
            # Build single-fold split indices
            _single_split = [(np.arange(len(train_feature_cv)), np.arange(len(train_feature_cv)), np.arange(len(val_feature)))]
            # Reassign training features (used in the fold loop)
            _cv_train_feature = train_feature_cv
            _cv_train_label = train_label_cv
            _cv_val_feature = val_feature
            _cv_val_label = val_label
            _use_single_split = True
        elif self.use_cluster_split:
            # Sequence-similarity cluster-based K-fold
            from utils.data import ClusterBasedKFold

            print("\n" + "="*70)
            print("Using sequence-similarity cluster-based K-fold cross-validation")
            print("="*70)

            # Cluster using raw sequences
            if hasattr(self, '_original_train_seqs'):
                train_seqs_for_kfold = self._original_train_seqs
            else:
                if self.encoding_type == 'onehot':
                    train_seqs_for_kfold = onehot2seq(train_feature)
                else:
                    raise ValueError("For non-one-hot encoding, raw training sequences must be kept during the cluster split")

            kf = ClusterBasedKFold(
                n_splits=self.n_folds,
                similarity_threshold=self.similarity_threshold,
                kmer_size=self.kmer_size,
                random_state=42,
                verbose=True
            )
            kf._fit(train_seqs_for_kfold)
            _use_single_split = False
        else:
            # Random K-fold split (legacy)
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            _use_single_split = False

        # Initialize metrics storage (per task type)
        if self.task_type == 'classification':
            metrics = {
                'val_accuracy': [],
                'val_precision': [],
                'val_recall': [],
                'val_f1': []
            }
        else:
            metrics = {
                'val_mse': [],
                'val_mae': [],
                'val_rmse': [],
                'val_pearson_r': [],
                'val_r2': []
            }

        fold_num = 1
        fold_metrics = defaultdict(list)
        fold_early_stop_epochs = []  # record early-stop epoch per fold
        fold_model_paths = []
        fold_val_scores = []  # for weighted ensembling (accuracy for classification, Pearson r for regression)

        # Single-fold mode uses the prepared split; multi-fold uses kf.split
        if _use_single_split:
            fold_splits = [(None, None)]  # placeholder, loop runs once
        elif _precomputed_splits is not None:
            fold_splits = _precomputed_splits  # ablation mode: precomputed manifest split
        else:
            fold_splits = kf.split(train_feature)

        for split_data in fold_splits:
            print(f"\n{'='*50}")
            print(f"Starting Fold {fold_num}")
            print(f"{'='*50}")
            _fold_start_time = time.perf_counter()  # ablation mode: record per-fold runtime

            if _use_single_split:
                fold_train_feature = _cv_train_feature
                fold_train_label = _cv_train_label
                fold_val_feature = _cv_val_feature
                fold_val_label = _cv_val_label
            else:
                train_index, val_index = split_data
                fold_train_feature = train_feature[train_index]
                fold_train_label = train_label[train_index]
                fold_val_feature = train_feature[val_index]
                fold_val_label = train_label[val_index]

            # Create datasets
            train_dataset = SequenceData(fold_train_feature, fold_train_label)
            val_dataset = SequenceData(fold_val_feature, fold_val_label)

            # Create DataLoaders
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
            # DataLoader for evaluating on the training set (used only when needed)
            train_eval_loader = DataLoader(SequenceData(fold_train_feature, fold_train_label),
                                          batch_size=self.batch_size, shuffle=False)

            # Create a fresh, independent model instance per fold
            model = ModelFactory.create(
                self.model_type,
                seq_len=self.seq_len,
                num_classes=self.num_classes if self.task_type == 'classification' else 1,  # num_classes for classification, 1 for regression
                dropout_rate=self.dropout_rate,
                conv_width_motif=self.conv_width_motif,
                n_heads=self.n_heads,
                conv_hidden=self.conv_hidden,
                motif_conv_hidden=self.motif_conv_hidden,
                vocab_size=self.vocab_size,
                task_type=self.task_type,  # pass task type
                ablation_variant=self.ablation_variant,  # ablation variant (effective for conv; 1dcnn ignores it)
            ).to(self.device)

            # Initialize weights
            for m in model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            # Ordinal regression: manually initialize or_bias to descending ordered values
            if self.task_type == 'classification':
                # Recursively find the ornet module in the model
                def init_or_bias(module):
                    if hasattr(module, 'ornet') and hasattr(module.ornet, 'or_bias'):
                        module.ornet.or_bias.data = torch.linspace(5.0, -5.0, self.num_classes - 1).to(self.device)
                    for child in module.children():
                        init_or_bias(child)
                init_or_bias(model)

            # Choose optimizer
            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")

            # Learning-rate scheduler
            lr_scheduler_patience = max(10, self.patience // 3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=lr_scheduler_patience)

            # Early Stopping
            fold_checkpoint = os.path.join(filename_sim, f'fold_{fold_num}')
            if not os.path.exists(fold_checkpoint):
                os.makedirs(fold_checkpoint)

            fold_model_path = os.path.join(fold_checkpoint, 'checkpoint.pth')
            fold_model_paths.append(fold_model_path)

            early_stopping = EarlyStopping_P(
                patience=self.patience,
                path=fold_model_path,
                stop_order='max'
            )

            for epoch in tqdm(range(0, self.epoch), desc=f"Fold {fold_num} Epochs"):
                # ============= Training =============
                model.train()
                train_epoch_loss = []
                for idx, (feature, label) in enumerate(train_dataloader, 0):
                    if self.vocab_size is not None:
                        feature = feature.to(torch.long).to(self.device)
                    else:
                        feature = feature.to(torch.float32).to(self.device).permute(0, 2, 1)

                    if self.task_type == 'classification':
                        label = label.to(torch.long).to(self.device)
                    else:
                        label = label.to(torch.float32).to(self.device)

                    # Mixup augmentation (regression only)
                    if self.use_mixup and self.task_type == 'regression':
                        from data.augmentation import mixup_batch
                        feature, label = mixup_batch(feature, label, alpha=self.mixup_alpha)

                    outputs = model(feature)
                    optimizer.zero_grad()

                    # Choose loss based on task type
                    if self.task_type == 'classification':
                        # Ordinal-regression loss: needs the model's or_bias parameter
                        or_bias = model.ornet.or_bias if hasattr(model, 'ornet') else None
                        if or_bias is not None:
                            loss = ordinal_regression_loss(outputs, label, or_bias)
                        else:
                            # Fallback to cross-entropy if the model has no ornet (should not happen)
                            loss = classification_loss(outputs, label)
                    else:
                        loss = regression_loss(outputs, label, self.l1_loss_weight)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                    optimizer.step()
                    train_epoch_loss.append(loss.item())

                avg_train_loss = sum(train_epoch_loss) / len(train_epoch_loss)

                # ============= Validation =============
                model.eval()
                val_pred = []
                val_true = []
                with torch.no_grad():
                    for idx, (feature, label) in enumerate(val_dataloader, 0):
                        if self.vocab_size is not None:
                            feature = feature.to(torch.long).to(self.device)
                        else:
                            feature = feature.to(torch.float32).to(self.device).permute(0, 2, 1)

                        if self.task_type == 'classification':
                            label = label.to(torch.long).to(self.device)
                        else:
                            label = label.to(torch.float32).to(self.device)

                        preds = model(feature)

                        if self.task_type == 'classification':
                            # Classification: convert ordinal logits to classes
                            pred_classes = logits2classes(preds)
                            val_true += label.cpu().tolist()
                            val_pred += pred_classes.cpu().tolist()
                        else:
                            # Regression: use predictions directly
                            val_true += label.cpu().tolist()
                            val_pred += preds.cpu().tolist()

                # [Fix #2: evaluate on the original scale] compute metrics per task type
                if self.task_type == 'classification':
                    val_metrics = calculate_classification_metrics(val_true, val_pred)
                    score_to_track = val_metrics['accuracy']  # classification tracks accuracy
                else:
                    # Regression: compute metrics
                    val_true_np = np.array(val_true)
                    val_pred_np = np.array(val_pred)
                    val_metrics = calculate_regression_metrics(val_true_np, val_pred_np)
                    score_to_track = val_metrics['pearson_r']  # regression tracks Pearson r (aligned with the Evo model)

                # ============= Compute training-set metrics (every epoch) =============
                model.eval()
                train_pred = []
                train_true = []
                with torch.no_grad():
                    for idx, (feature, label) in enumerate(train_eval_loader):
                        if self.vocab_size is not None:
                            feature = feature.to(torch.long).to(self.device)
                        else:
                            feature = feature.to(torch.float32).to(self.device).permute(0, 2, 1)

                        if self.task_type == 'classification':
                            label = label.to(torch.long).to(self.device)
                        else:
                            label = label.to(torch.float32).to(self.device)

                        preds = model(feature)

                        if self.task_type == 'classification':
                            # Classification: convert ordinal logits to classes
                            pred_classes = logits2classes(preds)
                            train_true += label.cpu().tolist()
                            train_pred += pred_classes.cpu().tolist()
                        else:
                            train_true += label.cpu().tolist()
                            train_pred += preds.cpu().tolist()

                if self.task_type == 'classification':
                    train_metrics = calculate_classification_metrics(train_true, train_pred)
                    print(f"\nFold {fold_num}, Epoch {epoch}")
                    print(f"  Train  - Loss: {avg_train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, "
                          f"F1: {train_metrics['f1']:.4f}")
                    print(f"  Val    - Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                          f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
                else:
                    # Regression: compute metrics
                    train_true_np = np.array(train_true)
                    train_pred_np = np.array(train_pred)
                    train_metrics = calculate_regression_metrics(train_true_np, train_pred_np)

                    print(f"\nFold {fold_num}, Epoch {epoch}")
                    print(f"  Train  - Loss: {avg_train_loss:.4f}, R^2: {train_metrics['r2']:.4f}, Pearson r: {train_metrics['pearson_r']:.4f}")
                    print(f"  Val    - R^2: {val_metrics['r2']:.4f}, Pearson r: {val_metrics['pearson_r']:.4f}, MAE: {val_metrics['mae']:.4f}")

                # Record validation metrics
                fold_metrics[fold_num].append(val_metrics)

                # ============= Early Stopping & Scheduler =============
                # Update scheduler first
                scheduler.step(score_to_track)

                # Then check early stopping
                early_stopping(score_to_track, model)
                if early_stopping.early_stop:
                    print(f'\nFold {fold_num}: Early Stopping triggered at epoch {epoch}.')
                    fold_early_stop_epochs.append(epoch)
                    break

            # Record the last epoch if early stopping was not triggered
            if not early_stopping.early_stop:
                fold_early_stop_epochs.append(self.epoch - 1)

            # Record this fold's best validation score (saved by early stopping)
            best_val_score = early_stopping.best_score
            fold_val_scores.append(best_val_score)

            # Accumulate metrics
            if self.task_type == 'classification':
                # For classification, keep the last epoch's metrics
                metrics['val_accuracy'].append(val_metrics['accuracy'])
                metrics['val_precision'].append(val_metrics['precision'])
                metrics['val_recall'].append(val_metrics['recall'])
                metrics['val_f1'].append(val_metrics['f1'])
            else:
                # Regression metrics
                metrics['val_mae'].append(val_metrics['mae'])
                metrics['val_mse'].append(val_metrics['mse'])
                metrics['val_rmse'].append(val_metrics['rmse'])
                metrics['val_pearson_r'].append(best_val_score)  # early-stopping monitored metric: Pearson r
                metrics['val_r2'].append(val_metrics['r2'])  # actual R^2 at the final epoch of this fold

            # Save per-fold metric curves
            self.plot_metrics_curves({fold_num: fold_metrics[fold_num]}, fold_checkpoint)

            # Ablation mode: write the ablation configuration into each fold checkpoint
            # (EarlyStopping_P stores only the raw state_dict)
            if _ablation_mode and os.path.exists(fold_model_path):
                _raw = torch.load(fold_model_path, map_location='cpu', weights_only=True)
                _sd = _raw['model_state_dict'] if isinstance(_raw, dict) and 'model_state_dict' in _raw else _raw
                torch.save({
                    'model_state_dict': _sd,
                    'ablation_variant': self.ablation_variant,
                    'model_type': self.model_type,
                    'task_type': self.task_type,
                    'conv_hidden': self.conv_hidden,
                    'motif_conv_hidden': self.motif_conv_hidden,
                    'n_heads': self.n_heads,
                    'conv_width_motif': self.conv_width_motif,
                    'num_classes': self.num_classes,
                    'fold_val_score': best_val_score,
                }, fold_model_path)

            # Ablation mode: write per-fold detailed metrics to fold_details.jsonl
            # (consumed by the aggregator)
            if _ablation_mode:
                _fold_runtime = time.perf_counter() - _fold_start_time
                _ep_metrics = fold_metrics.get(fold_num, [])
                _record = {'fold': fold_num, 'split': 'val', 'runtime_sec': _fold_runtime}
                if _ep_metrics:
                    if self.task_type == 'classification':
                        _scores = [m.get('accuracy', float('nan')) for m in _ep_metrics]
                    else:
                        _scores = [m.get('pearson_r', float('nan')) for m in _ep_metrics]
                    _best_idx = int(np.nanargmax(_scores)) if len(_scores) > 0 else 0
                    _best = _ep_metrics[_best_idx]
                    _record['best_epoch'] = _best_idx
                    if self.task_type == 'classification':
                        _record.update({k: _best.get(k) for k in ('accuracy', 'precision', 'recall', 'f1')})
                    else:
                        _record.update({k: _best.get(k) for k in ('pearson_r', 'spearman_r', 'rmse', 'mae', 'r2')})
                _fd_path = os.path.join(filename_sim, 'fold_details.jsonl')
                with open(_fd_path, 'a') as _f:
                    _f.write(json.dumps(_record, default=float) + '\n')

            # Release GPU memory
            del model, optimizer, scheduler
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            fold_num += 1

        # Save all fold model paths and weights (used by test())
        self.fold_model_paths = fold_model_paths
        self.fold_val_scores = fold_val_scores  # stored as an instance attribute (generic name)

        # Save combined metric curves across all folds
        self.plot_metrics_curves(fold_metrics, filename_sim)

        # Compute mean and std of cross-validation results
        results = {}
        for metric in metrics:
            if len(metrics[metric]) > 0:
                mean = np.mean(metrics[metric])
                std = np.std(metrics[metric])
                results[metric] = (mean, std)
            else:
                results[metric] = (0.0, 0.0)

        print("\n" + "="*70)
        print("5-Fold Cross-Validation Results (Mean +/- Std):")
        print("="*70)
        if self.task_type == 'classification':
            print("Validation Set:")
            print(f"  Accuracy:  {results['val_accuracy'][0]:.4f} +/- {results['val_accuracy'][1]:.4f}")
            print(f"  Precision: {results['val_precision'][0]:.4f} +/- {results['val_precision'][1]:.4f}")
            print(f"  Recall:    {results['val_recall'][0]:.4f} +/- {results['val_recall'][1]:.4f}")
            print(f"  F1:        {results['val_f1'][0]:.4f} +/- {results['val_f1'][1]:.4f}")
        else:
            print("Validation Set:")
            print(f"  MAE:       {results['val_mae'][0]:.4f} +/- {results['val_mae'][1]:.4f}")
            print(f"  MSE:       {results['val_mse'][0]:.4f} +/- {results['val_mse'][1]:.4f}")
            print(f"  RMSE:      {results['val_rmse'][0]:.4f} +/- {results['val_rmse'][1]:.4f}")
            print(f"  Pearson r: {results['val_pearson_r'][0]:.4f} +/- {results['val_pearson_r'][1]:.4f}")
            print(f"  R^2:       {results['val_r2'][0]:.4f} +/- {results['val_r2'][1]:.4f}")
        print("="*70)

        # Save cross-validation results
        with open(os.path.join(filename_sim, "cv_metrics_summary.txt"), "w") as f:
            f.write("5-Fold Cross-Validation Results (Mean +/- Std):\n")
            f.write("="*70 + "\n")
            f.write(f"Task Type: {self.task_type}\n")
            f.write("\nValidation Set:\n")
            if self.task_type == 'classification':
                f.write(f"  Accuracy:  {results['val_accuracy'][0]:.4f} +/- {results['val_accuracy'][1]:.4f}\n")
                f.write(f"  Precision: {results['val_precision'][0]:.4f} +/- {results['val_precision'][1]:.4f}\n")
                f.write(f"  Recall:    {results['val_recall'][0]:.4f} +/- {results['val_recall'][1]:.4f}\n")
                f.write(f"  F1:        {results['val_f1'][0]:.4f} +/- {results['val_f1'][1]:.4f}\n")
            else:
                f.write(f"  MAE:       {results['val_mae'][0]:.4f} +/- {results['val_mae'][1]:.4f}\n")
                f.write(f"  MSE:       {results['val_mse'][0]:.4f} +/- {results['val_mse'][1]:.4f}\n")
                f.write(f"  RMSE:      {results['val_rmse'][0]:.4f} +/- {results['val_rmse'][1]:.4f}\n")
                f.write(f"  Pearson r: {results['val_pearson_r'][0]:.4f} +/- {results['val_pearson_r'][1]:.4f}\n")
                f.write(f"  R^2:       {results['val_r2'][0]:.4f} +/- {results['val_r2'][1]:.4f}\n")
            f.write("="*70 + "\n\n")

            # Save per-fold early-stop info
            f.write("Early Stopping Information:\n")
            score_name = 'accuracy' if self.task_type == 'classification' else 'Pearson r'
            for i, epoch in enumerate(fold_early_stop_epochs, 1):
                f.write(f"  Fold {i}: stopped at epoch {epoch}, best val {score_name} = {fold_val_scores[i-1]:.4f}\n")

        print(f"\nCross-validation complete! Results saved to: {filename_sim}")
        print("Use the test() method to evaluate on the test set.")

        # Return value: results, test_feature, test_label
        return self.test_feature, self.test_label

    def test(self, save_dir=None, return_metrics=False, num_bins=5, label_min=0.0, label_max=1.0,
             return_predictions=False):
        """
        Predict and evaluate the test set using the ensemble.

        save_dir: directory to save test results
        return_metrics: whether to return the metrics dict
        num_bins: number of label bins (regression only)
        label_min: minimum label value (regression only)
        label_max: maximum label value (regression only)
        return_predictions: if True, return (per_fold_predictions, ensemble_predictions, test_seqs)
            for the ablation mode to write the canonical test_results.csv. For classification,
            ensemble_predictions is an array of class labels.
        """
        if not hasattr(self, 'test_feature') or not hasattr(self, 'fold_model_paths'):
            raise ValueError("Please run train() first!")

        if save_dir is None:
            # Determine default save directory
            if self.checkpoint_root is not None:
                save_dir = os.path.join(self.checkpoint_root, 'test_results')
            elif self.fold_model_paths and len(self.fold_model_paths) > 0:
                # Infer checkpoint dir from fold_model_paths
                first_fold_path = self.fold_model_paths[0]
                model_dir = os.path.dirname(os.path.dirname(first_fold_path))
                save_dir = os.path.join(model_dir, 'test_results')
            else:
                raise ValueError("Cannot determine default save directory: both checkpoint_root and fold_model_paths are empty")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        test_feature = self.test_feature
        test_label = self.test_label

        # Build output paths
        path_seq_save = os.path.join(save_dir, 'test_sequences.txt')
        path_pred_save = os.path.join(save_dir, 'test_predictions.txt')
        path_csv_save = os.path.join(save_dir, 'test_results.csv')

        device = self.device

        # Recover sequences according to encoding
        if self.encoding_type == 'onehot':
            test_seqs = onehot2seq(test_feature)
        else:
            from data.sequence_encoding import encode_sequences
            _, stoi, itos, _ = encode_sequences(['A'], self.encoding_type)
            test_seqs = []
            test_feature_np = test_feature.numpy() if torch.is_tensor(test_feature) else test_feature

            for seq_tokens in test_feature_np:
                seq = ''
                # Filter out padding tokens (0)
                seq_tokens = [t for t in seq_tokens if t != 0]

                if self.encoding_type == 'single':
                    for token in seq_tokens:
                        seq += itos.get(token, 'N')
                elif self.encoding_type == 'pairs':
                    for i, token in enumerate(seq_tokens):
                        pair = itos.get(token, 'NN')
                        if i == 0:
                            seq += pair
                        else:
                            seq += pair[1]
                elif self.encoding_type == 'triples':
                    for i, token in enumerate(seq_tokens):
                        triple = itos.get(token, 'NNN')
                        if i == 0:
                            seq += triple
                        else:
                            seq += triple[-1]
                else:
                    seq = 'N' * len(seq_tokens)
                test_seqs.append(seq)

        # Build the test DataLoader
        test_dataset = SequenceData(test_feature, test_label)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # Ensemble prediction
        print(f"\n{'='*70}")
        print(f"Ensemble prediction using {len(self.fold_model_paths)} fold models")
        print(f"Task type: {self.task_type}")
        print(f"Ensemble method: {self.ensemble_method}")
        print(f"{'='*70}")

        all_fold_predictions = []

        for fold_idx, fold_model_path in enumerate(self.fold_model_paths, 1):
            print(f"Loading Fold {fold_idx} model...")

            # Create the model and load weights
            fold_model = ModelFactory.create(
                self.model_type,
                seq_len=self.seq_len,
                num_classes=self.num_classes if self.task_type == 'classification' else 1,
                dropout_rate=self.dropout_rate,
                conv_width_motif=self.conv_width_motif,
                n_heads=self.n_heads,
                conv_hidden=self.conv_hidden,
                motif_conv_hidden=self.motif_conv_hidden,
                vocab_size=self.vocab_size,
                task_type=self.task_type,
                ablation_variant=self.ablation_variant,
            ).to(device)

            # Support both raw state_dict (legacy checkpoints) and dict (ablation checkpoints)
            _fold_ckpt = torch.load(fold_model_path, weights_only=True)
            _fold_sd = _fold_ckpt['model_state_dict'] if isinstance(_fold_ckpt, dict) and 'model_state_dict' in _fold_ckpt else _fold_ckpt
            fold_model.load_state_dict(_fold_sd)
            fold_model.eval()

            # Predict
            fold_predictions = []
            with torch.no_grad():
                for feature, label in test_dataloader:
                    if self.vocab_size is not None:
                        feature = feature.to(torch.long).to(device)
                    else:
                        feature = feature.to(torch.float32).to(device).permute(0, 2, 1)

                    outputs = fold_model(feature)

                    if self.task_type == 'classification':
                        # Classification: keep logits for later voting ensemble
                        fold_predictions.append(outputs.cpu().numpy())
                    else:
                        # Regression: keep prediction values
                        fold_predictions.extend(outputs.detach().cpu().numpy().flatten())

            # Concatenate per-batch predictions into a single array
            if self.task_type == 'classification':
                fold_predictions = np.concatenate(fold_predictions, axis=0)
            else:
                fold_predictions = np.array(fold_predictions)

            all_fold_predictions.append(fold_predictions)

            # Release memory
            del fold_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Get ground-truth labels (handle both tensor and numpy array)
        if torch.is_tensor(test_label):
            test_true_labels = test_label.cpu().numpy()
        else:
            test_true_labels = test_label  # already a numpy array

        # Ensemble prediction and metric computation per task type
        if self.task_type == 'classification':
            # ========== Classification mode (ordinal regression) ==========
            # Ensemble: vote across folds
            all_fold_predictions = np.array(all_fold_predictions)  # shape: (5, num_samples, num_classes-1)

            # Helper: convert ordinal-regression logits to classes
            def numpy_logits2classes(logits):
                """numpy version of logits2classes"""
                probs = 1 / (1 + np.exp(-logits))  # sigmoid
                n_samples = probs.shape[0]
                # Build the difference matrix
                x = np.concatenate([
                    np.ones((n_samples, 1)),
                    probs,
                    np.zeros((n_samples, 1))
                ], axis=1)
                x = x[:, :-1] - x[:, 1:]
                classes = np.argmax(x, axis=1)
                return classes

            if self.ensemble_method == 'mean':
                # Average logits then derive predicted classes
                avg_logits = np.mean(all_fold_predictions, axis=0)
                test_pred_classes = numpy_logits2classes(avg_logits)
                print("Ensemble via logit averaging")
            elif self.ensemble_method == 'weighted':
                # Use validation accuracy as weights
                weights = np.array(self.fold_val_scores)
                weights = weights / np.sum(weights)
                weighted_logits = np.average(all_fold_predictions, axis=0, weights=weights)
                test_pred_classes = numpy_logits2classes(weighted_logits)
                print(f"Ensemble via weighted averaging, weights: {weights}")
            elif self.ensemble_method == 'median':
                # Take the median of logits then derive predicted classes
                median_logits = np.median(all_fold_predictions, axis=0)
                test_pred_classes = numpy_logits2classes(median_logits)
                print("Ensemble via logit median")
            elif self.ensemble_method == 'vote':
                # Hard voting: each fold casts one vote
                fold_preds = numpy_logits2classes(all_fold_predictions.transpose(1, 0, 2).reshape(-1, all_fold_predictions.shape[2]))
                fold_preds = fold_preds.reshape(all_fold_predictions.shape[1], all_fold_predictions.shape[0])
                test_pred_classes = []
                for i in range(fold_preds.shape[0]):
                    votes = fold_preds[i, :]
                    test_pred_classes.append(np.bincount(votes.astype(int)).argmax())
                test_pred_classes = np.array(test_pred_classes)
                print("Ensemble via hard voting")
            else:
                raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")

            # Compute classification metrics and probabilities
            test_metrics = calculate_classification_metrics(test_true_labels, test_pred_classes)

            # Compute per-sample prediction probabilities (from averaged logits)
            avg_logits = np.mean(all_fold_predictions, axis=0)
            probs = 1 / (1 + np.exp(-avg_logits))  # sigmoid
            n_samples = probs.shape[0]
            # Build class probabilities
            x = np.concatenate([
                np.ones((n_samples, 1)),
                probs,
                np.zeros((n_samples, 1))
            ], axis=1)
            x = x[:, :-1] - x[:, 1:]
            # Normalize to a probability distribution
            test_probs = x / x.sum(axis=1, keepdims=True)

            print("\n" + "="*70)
            print("Ensemble test-set metrics:")
            print("="*70)
            print(f"  Accuracy:  {test_metrics['accuracy']:.6f}")
            print(f"  Precision: {test_metrics['precision']:.6f}")
            print(f"  Recall:    {test_metrics['recall']:.6f}")
            print(f"  F1:        {test_metrics['f1']:.6f}")
            print("="*70)

            # Save metrics to file
            path_metrics_save = os.path.join(save_dir, 'test_metrics.txt')
            with open(path_metrics_save, 'w') as f:
                f.write(f"Ensemble method: {self.ensemble_method}\n")
                f.write(f"Task type: {self.task_type}\n")
                f.write("="*70 + "\n")
                f.write("Test Set Metrics:\n")
                f.write(f"  Accuracy:  {test_metrics['accuracy']:.6f}\n")
                f.write(f"  Precision: {test_metrics['precision']:.6f}\n")
                f.write(f"  Recall:    {test_metrics['recall']:.6f}\n")
                f.write(f"  F1:        {test_metrics['f1']:.6f}\n")
                f.write("="*70 + "\n\n")

                if self.ensemble_method == 'weighted':
                    f.write("Per-fold weights (based on validation Accuracy):\n")
                    for i, (w, acc) in enumerate(zip(weights, self.fold_val_scores), 1):
                        f.write(f"  Fold {i}: weight={w:.4f}, val_Accuracy={acc:.4f}\n")

            # Save sequences
            with open(path_seq_save, 'w') as f:
                for i, seq in enumerate(test_seqs):
                    f.write(f'>{i}\n{seq}\n')

            # Save predicted classes and probabilities
            with open(path_pred_save, 'w') as f:
                for i, (pred, prob) in enumerate(zip(test_pred_classes, test_probs)):
                    prob_str = ', '.join([f'prob_class_{c}={prob[c]:.6f}' for c in range(self.num_classes)])
                    f.write(f'{pred} ({prob_str})\n')

            # Save detailed results to CSV (including probabilities)
            with open(path_csv_save, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['Sequence', 'Predicted_Class', 'Ground_Truth']
                for c in range(self.num_classes):
                    header.append(f'Probability_Class_{c}')
                for i in range(1, len(self.fold_model_paths)+1):
                    for k in range(self.num_classes - 1):
                        header.extend([f'Fold_{i}_Logits_{k}'])
                writer.writerow(header)

                for idx in range(len(test_seqs)):
                    seq = test_seqs[idx]
                    true_label = int(test_true_labels[idx])
                    pred_label = int(test_pred_classes[idx])

                    row_data = [seq, pred_label, true_label]
                    for c in range(self.num_classes):
                        row_data.append(f'{test_probs[idx][c]:.6f}')
                    # Append per-fold logits
                    for fold_logits in all_fold_predictions:
                        for k in range(self.num_classes - 1):
                            row_data.append(f'{fold_logits[idx][k]:.6f}')

                    writer.writerow(row_data)

        else:
            # ========== Regression mode ==========
            # Ensemble prediction
            all_fold_predictions = np.array(all_fold_predictions)  # shape: (5, num_samples)

            if self.ensemble_method == 'mean':
                test_exp_pred = np.mean(all_fold_predictions, axis=0)
                print("Ensemble via simple averaging")
            elif self.ensemble_method == 'weighted':
                weights = np.array(self.fold_val_scores)
                weights = weights / np.sum(weights)
                test_exp_pred = np.average(all_fold_predictions, axis=0, weights=weights)
                print(f"Ensemble via weighted averaging, weights: {weights}")
            elif self.ensemble_method == 'median':
                test_exp_pred = np.median(all_fold_predictions, axis=0)
                print("Ensemble via median")
            else:
                raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")

            # Compute test-set metrics
            test_metrics = calculate_regression_metrics(test_true_labels, test_exp_pred)

            # Print metrics
            print("\n" + "="*70)
            print("Ensemble test-set metrics:")
            print("="*70)
            print(f"  R^2:       {test_metrics['r2']:.6f}")
            print(f"  Pearson r: {test_metrics['pearson_r']:.6f}")
            print(f"  MSE:       {test_metrics['mse']:.6f}")
            print(f"  MAE:       {test_metrics['mae']:.6f}")
            print(f"  RMSE:      {test_metrics['rmse']:.6f}")
            print("="*70)

            # Compute R^2 stratified by label bins
            bins_info = calculate_r2_by_bins(test_true_labels, test_exp_pred,
                                             num_bins=num_bins,
                                             min_val=label_min,
                                             max_val=label_max)

            # Print per-bin R^2
            print("\n" + "="*70)
            print(f"R^2 by label bin ({num_bins} bins):")
            print("="*70)
            print(f"{'Bin':<20} {'Samples':<10} {'R^2':<12} {'MSE':<12} {'MAE':<12}")
            print("-" * 70)
            for bin_info in bins_info:
                r2_s = 'N/A' if np.isnan(bin_info['r2']) else f"{bin_info['r2']:.4f}"
                mse_s = 'N/A' if np.isnan(bin_info['mse']) else f"{bin_info['mse']:.4f}"
                mae_s = 'N/A' if np.isnan(bin_info['mae']) else f"{bin_info['mae']:.4f}"
                print(f"{bin_info['bin_range']:<20} {bin_info['sample_count']:<10} "
                      f"{r2_s:<12} {mse_s:<12} {mae_s:<12}")
            print("="*70)

            # =====================================================================
            # Sequence-similarity stratified performance report (generalization assessment)
            # =====================================================================
            # For each test sequence, compute the Hamming distance to the nearest training sequence
            if hasattr(self, '_train_seqs_for_similarity'):
                train_seqs_set = set(self._train_seqs_for_similarity)
                min_hamming_dists = []
                for test_seq in test_seqs:
                    # If the test sequence happens to be in the training set (should not happen), mark as 0
                    if test_seq in train_seqs_set:
                        min_hamming_dists.append(0)
                        continue
                    # Compute minimum Hamming distance
                    min_dist = len(test_seq)  # initialize to sequence length
                    for train_seq in train_seqs_set:
                        dist = sum(1 for a, b in zip(test_seq, train_seq) if a != b)
                        if dist < min_dist:
                            min_dist = dist
                            if min_dist == 1:
                                break  # early termination
                    min_hamming_dists.append(min_dist)

                min_hamming_dists = np.array(min_hamming_dists)
                # Similarity strata: identical, 1 mismatch, 2, 3-4, 5+
                sim_bins = [
                    (0, 0, "Identical (d=0)"),
                    (1, 1, "Near-identical (d=1)"),
                    (2, 2, "Highly similar (d=2)"),
                    (3, 4, "Moderately similar (d=3-4)"),
                    (5, 999, "Dissimilar (d>=5)"),
                ]

                print("\n" + "="*70)
                print("Sequence-similarity stratified performance report (binned by min Hamming distance to training set):")
                print("="*70)
                print(f"{'Similarity stratum':<30} {'Samples':<10} {'R^2':<12} {'Pearson r':<12} {'MAE':<12}")
                print("-" * 70)

                sim_stratified_results = []
                for lo, hi, label in sim_bins:
                    mask = (min_hamming_dists >= lo) & (min_hamming_dists <= hi)
                    n_bin = mask.sum()
                    if n_bin >= 3:
                        bin_true = test_true_labels[mask]
                        bin_pred = test_exp_pred[mask]
                        bin_metrics = calculate_regression_metrics(bin_true, bin_pred)
                        r2_str = f"{bin_metrics['r2']:.4f}"
                        pr_str = f"{bin_metrics['pearson_r']:.4f}"
                        mae_str = f"{bin_metrics['mae']:.4f}"
                    elif n_bin > 0:
                        r2_str = "N/A (<3)"
                        pr_str = "N/A (<3)"
                        mae_str = "N/A (<3)"
                    else:
                        r2_str = "N/A (0)"
                        pr_str = "N/A (0)"
                        mae_str = "N/A (0)"

                    print(f"{label:<30} {n_bin:<10} {r2_str:<12} {pr_str:<12} {mae_str:<12}")
                    sim_stratified_results.append({
                        'similarity_bin': label,
                        'n_samples': int(n_bin),
                        'r2': r2_str,
                        'pearson_r': pr_str,
                        'mae': mae_str,
                    })

                # Extra: summarize "has neighbor" vs "no neighbor" (d<=1 vs d>=3)
                mask_near = min_hamming_dists <= 1
                mask_far = min_hamming_dists >= 3
                if mask_near.sum() >= 3 and mask_far.sum() >= 3:
                    near_metrics = calculate_regression_metrics(
                        test_true_labels[mask_near], test_exp_pred[mask_near])
                    far_metrics = calculate_regression_metrics(
                        test_true_labels[mask_far], test_exp_pred[mask_far])
                    generalization_gap = near_metrics['pearson_r'] - far_metrics['pearson_r']
                    print("-" * 70)
                    print(f"{'Has neighbor (d<=1)':<30} {mask_near.sum():<10} "
                          f"{near_metrics['r2']:.4f}       {near_metrics['pearson_r']:.4f}       {near_metrics['mae']:.4f}")
                    print(f"{'No neighbor (d>=3)':<30} {mask_far.sum():<10} "
                          f"{far_metrics['r2']:.4f}       {far_metrics['pearson_r']:.4f}       {far_metrics['mae']:.4f}")
                    print(f"{'Generalization gap (delta r)':<30} {'':<10} "
                          f"{'':<12} {generalization_gap:+.4f}      {'':<12}")
                print("="*70)

                # Save similarity-stratified results
                sim_csv_path = os.path.join(save_dir, 'similarity_stratified_metrics.csv')
                with open(sim_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['similarity_bin', 'n_samples', 'r2', 'pearson_r', 'mae'])
                    for r in sim_stratified_results:
                        writer.writerow([r['similarity_bin'], r['n_samples'],
                                        r['r2'], r['pearson_r'], r['mae']])
                print(f"Similarity-stratified results saved to: {sim_csv_path}")
            else:
                print("\n[Note] Training-sequence info not stored; skipping similarity-stratified report. "
                      "Set self._train_seqs_for_similarity inside train() to enable it.")

            # Save metrics to file
            path_metrics_save = os.path.join(save_dir, 'test_metrics.txt')
            with open(path_metrics_save, 'w') as f:
                f.write(f"Ensemble method: {self.ensemble_method}\n")
                f.write(f"Task type: {self.task_type}\n")
                f.write("="*70 + "\n")
                f.write(f"  R^2:       {test_metrics['r2']:.6f}\n")
                f.write(f"  Pearson r: {test_metrics['pearson_r']:.6f}\n")
                f.write(f"  MSE:       {test_metrics['mse']:.6f}\n")
                f.write(f"  MAE:       {test_metrics['mae']:.6f}\n")
                f.write(f"  RMSE:      {test_metrics['rmse']:.6f}\n")
                f.write("="*70 + "\n\n")

                if self.ensemble_method == 'weighted':
                    f.write("Per-fold weights (based on validation R^2):\n")
                    for i, (w, r2) in enumerate(zip(weights, self.fold_val_scores), 1):
                        f.write(f"  Fold {i}: weight={w:.4f}, val_R^2={r2:.4f}\n")

            # Save sequences
            with open(path_seq_save, 'w') as f:
                for i, seq in enumerate(test_seqs):
                    f.write(f'>{i}\n{seq}\n')

            # Save predictions
            with open(path_pred_save, 'w') as f:
                for pred in test_exp_pred:
                    f.write(f'{pred:.6f}\n')

            # Save detailed results to CSV
            with open(path_csv_save, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['Sequence',
                         'Ensemble_Prediction', 'Ground_Truth'] + \
                         [f'Fold_{i}_Prediction' for i in range(1, len(self.fold_model_paths)+1)]
                writer.writerow(header)

                for idx in range(len(test_seqs)):
                    seq = test_seqs[idx]
                    row_data = [seq,
                                f'{test_exp_pred[idx]:.6f}', f'{test_true_labels[idx]:.6f}']
                    for fold_preds in all_fold_predictions:
                        row_data.append(f'{fold_preds[idx]:.6f}')

                    writer.writerow(row_data)

        # Common footer
        print(f"\nTest results saved to: {save_dir}")
        print(f"  - Sequence file: {path_seq_save}")
        print(f"  - Prediction file: {path_pred_save}")
        print(f"  - Detailed CSV: {path_csv_save}")
        print(f"  - Metrics summary: {path_metrics_save}")

        if return_predictions:
            # Ablation mode: return per-fold predictions + ensemble predictions + test
            # sequences for the caller to write the canonical CSV
            if self.task_type == 'classification':
                # Return class-1 probabilities (sigmoid of per-fold logits + ensemble
                # probabilities) for downstream AUROC/AUPRC computation
                _logits = np.asarray(all_fold_predictions)  # (n_folds, n_samples, n_classes-1)
                _per_fold = [1.0 / (1.0 + np.exp(-_logits[k, :, 0])) for k in range(_logits.shape[0])]
                _ens = test_probs[:, 1] if test_probs.shape[1] >= 2 else test_probs[:, 0]
                return _per_fold, _ens, test_seqs
            else:
                return all_fold_predictions, test_exp_pred, test_seqs

        if return_metrics:
            return test_metrics
        else:
            if self.task_type == 'classification':
                return test_metrics['accuracy']  # classification returns accuracy
            else:
                return test_metrics['r2']  # regression returns R^2

    def predict(self, seqs, save_path=None, return_probs=False):
        """
        Predict on new sequences using the ensemble model.

        seqs: list of sequences to predict
        save_path: optional path to save predictions
        return_probs: whether to return probabilities (classification only)

        Returns:
            - Regression: array of predictions
            - Classification:
                - return_probs=False: array of predicted classes
                - return_probs=True: (predicted_classes, probabilities)
        """
        # Process input sequences
        if self.encoding_type == 'onehot':
            feature = seq2onehot(seqs)
            feature = torch.tensor(feature, dtype=torch.float32)
        else:
            from data.sequence_encoding import encode_sequences
            encoded_seqs, _, _, _ = encode_sequences(seqs, self.encoding_type)
            max_len = max(len(seq) for seq in encoded_seqs)
            padded_seqs = []
            for seq in encoded_seqs:
                if len(seq) < max_len:
                    padding_token = 0
                    padding = np.full(max_len - len(seq), padding_token)
                    seq = np.concatenate([seq, padding])
                padded_seqs.append(seq)
            feature = torch.tensor(np.array(padded_seqs), dtype=torch.long)

        # Build DataLoader
        predict_dataset = TestData(feature)
        predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=self.batch_size, shuffle=False)

        device = self.device
        all_fold_predictions = []

        print(f"Predicting with {len(self.fold_model_paths)} fold models...")
        print(f"Task type: {self.task_type}")

        # Each fold predicts
        for fold_idx, fold_model_path in enumerate(self.fold_model_paths, 1):
            fold_model = ModelFactory.create(
                self.model_type,
                seq_len=self.seq_len,
                num_classes=self.num_classes if self.task_type == 'classification' else 1,
                dropout_rate=self.dropout_rate,
                conv_width_motif=self.conv_width_motif,
                n_heads=self.n_heads,
                conv_hidden=self.conv_hidden,
                motif_conv_hidden=self.motif_conv_hidden,
                vocab_size=self.vocab_size,
                task_type=self.task_type,
                ablation_variant=self.ablation_variant,
            ).to(device)

            # Support both raw state_dict (legacy checkpoints) and dict (ablation checkpoints)
            _fold_ckpt = torch.load(fold_model_path, weights_only=True)
            _fold_sd = _fold_ckpt['model_state_dict'] if isinstance(_fold_ckpt, dict) and 'model_state_dict' in _fold_ckpt else _fold_ckpt
            fold_model.load_state_dict(_fold_sd)
            fold_model.eval()

            fold_predictions = []
            with torch.no_grad():
                for batch_feature in predict_dataloader:
                    if self.vocab_size is not None:
                        batch_feature = batch_feature.to(torch.long).to(device)
                    else:
                        batch_feature = batch_feature.to(torch.float32).to(device).permute(0, 2, 1)

                    outputs = fold_model(batch_feature)

                    if self.task_type == 'classification':
                        # Classification: keep logits
                        fold_predictions.append(outputs.cpu().numpy())
                    else:
                        # Regression: keep prediction values
                        fold_predictions.extend(outputs.detach().cpu().numpy().flatten())

            if self.task_type == 'classification':
                # Classification: concatenate per-batch logits into a single array
                fold_predictions = np.concatenate(fold_predictions, axis=0)
            all_fold_predictions.append(fold_predictions)

            del fold_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Ensemble prediction
        all_fold_predictions = np.array(all_fold_predictions)

        if self.task_type == 'classification':
            # ========== Classification mode (ordinal regression) ==========
            # Helper: convert ordinal-regression logits to classes
            def numpy_logits2classes(logits):
                """numpy version of logits2classes"""
                probs = 1 / (1 + np.exp(-logits))  # sigmoid
                n_samples = probs.shape[0]
                # Build the difference matrix
                x = np.concatenate([
                    np.ones((n_samples, 1)),
                    probs,
                    np.zeros((n_samples, 1))
                ], axis=1)
                x = x[:, :-1] - x[:, 1:]
                classes = np.argmax(x, axis=1)
                return classes

            if self.ensemble_method == 'mean':
                avg_logits = np.mean(all_fold_predictions, axis=0)
                predictions = numpy_logits2classes(avg_logits)
                probs = 1 / (1 + np.exp(-avg_logits))  # sigmoid
                print("Ensemble via logit averaging")
            elif self.ensemble_method == 'weighted':
                weights = np.array(self.fold_val_scores)
                weights = weights / np.sum(weights)
                weighted_logits = np.average(all_fold_predictions, axis=0, weights=weights)
                predictions = numpy_logits2classes(weighted_logits)
                probs = 1 / (1 + np.exp(-weighted_logits))  # sigmoid
                print(f"Ensemble via weighted averaging, weights: {weights}")
            elif self.ensemble_method == 'median':
                median_logits = np.median(all_fold_predictions, axis=0)
                predictions = numpy_logits2classes(median_logits)
                probs = 1 / (1 + np.exp(-np.mean(all_fold_predictions, axis=0)))  # probabilities from averaged logits
                print("Ensemble via logit median")
            elif self.ensemble_method == 'vote':
                # Hard voting: each fold casts one vote
                fold_preds = numpy_logits2classes(all_fold_predictions.transpose(1, 0, 2).reshape(-1, all_fold_predictions.shape[2]))
                fold_preds = fold_preds.reshape(all_fold_predictions.shape[1], all_fold_predictions.shape[0])
                predictions = []
                for i in range(fold_preds.shape[0]):
                    votes = fold_preds[i, :]
                    predictions.append(np.bincount(votes.astype(int)).argmax())
                predictions = np.array(predictions)
                probs = 1 / (1 + np.exp(-np.mean(all_fold_predictions, axis=0)))
                print("Ensemble via hard voting")
            else:
                raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")

            # Compute class probabilities (from ordinal-regression logits)
            n_samples = probs.shape[0]
            x = np.concatenate([
                np.ones((n_samples, 1)),
                probs,
                np.zeros((n_samples, 1))
            ], axis=1)
            x = x[:, :-1] - x[:, 1:]
            probs = x / x.sum(axis=1, keepdims=True)  # normalize

            # Optionally save predictions
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # CSV header includes probabilities, multi-class supported
                    header = ['Sequence', 'Predicted_Class']
                    for c in range(self.num_classes):
                        header.append(f'Probability_Class_{c}')
                    for i in range(1, len(self.fold_model_paths)+1):
                        for k in range(self.num_classes - 1):
                            header.extend([f'Fold_{i}_Logits_{k}'])
                    writer.writerow(header)

                    for idx, seq in enumerate(seqs):
                        row = [seq, int(predictions[idx])]
                        for c in range(self.num_classes):
                            row.append(f'{probs[idx][c]:.6f}')
                        # Append per-fold logits
                        for fold_logits in all_fold_predictions:
                            for k in range(self.num_classes - 1):
                                row.append(f'{fold_logits[idx][k]:.6f}')
                        writer.writerow(row)

                print(f"Predictions saved to: {save_path}")

            # Return results
            if return_probs:
                return predictions, probs
            return predictions

        else:
            # ========== Regression mode ==========
            if self.ensemble_method == 'mean':
                predictions = np.mean(all_fold_predictions, axis=0)
            elif self.ensemble_method == 'weighted':
                weights = np.array(self.fold_val_scores)
                weights = weights / np.sum(weights)
                predictions = np.average(all_fold_predictions, axis=0, weights=weights)
            elif self.ensemble_method == 'median':
                predictions = np.median(all_fold_predictions, axis=0)
            else:
                raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")

            # Optionally save predictions
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['Sequence',
                             'Ensemble_Prediction'] + \
                             [f'Fold_{i}_Prediction' for i in range(1, len(self.fold_model_paths)+1)]
                    writer.writerow(header)

                    for idx, seq in enumerate(seqs):
                        row = [seq, f'{predictions[idx]:.6f}']
                        for fold_preds in all_fold_predictions:
                            row.append(f'{fold_preds[idx]:.6f}')
                        writer.writerow(row)

                print(f"Predictions saved to: {save_path}")

            return predictions

