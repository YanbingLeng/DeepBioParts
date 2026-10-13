"""
Position importance computation for DeepBioParts predictor models.

This module provides computation utilities for analyzing per-position importance
scores using attention weights and gradient-based saliency. It contains no
plotting logic; visualization functions remain in sequence_evaluation/.

Functions:
    load_model_config: Load model configuration from a checkpoint directory.
    load_model_from_checkpoint: Load a trained model from a checkpoint file.
    get_attention_weights_from_model: Extract attention-based importance weights.
    compute_gradient_based_importance: Compute gradient saliency importance.
    compute_position_importance: Aggregate per-position importance statistics.
    save_position_importance_report: Persist importance reports as CSV files.
    evaluate_position_importance: Run the full evaluation pipeline and return results.
"""

import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure parent directories are importable so ModelFactory / seq2onehot resolve.
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predictor import ModelFactory  # noqa: E402
try:
    from data.encoding import seq2onehot  # noqa: E402
except ImportError:
    from utils.data import seq2onehot  # noqa: E402


# ======================================================================
# Model loading helpers
# ======================================================================

def load_model_config(checkpoint_dir: str) -> Dict[str, Any]:
    """Load model configuration from a checkpoint directory.

    Reads ``best_params.pkl`` from the checkpoint directory and infers
    model type, encoding, biopart type, and task type from the directory
    name.

    Args:
        checkpoint_dir: Path to a checkpoint directory that contains
            ``best_params.pkl``.

    Returns:
        A dictionary with keys ``model_type``, ``encoding_type``,
        ``biopart_type``, ``seq_len``, ``task_type`` plus all values
        stored in ``best_params.pkl``.

    Raises:
        FileNotFoundError: If ``best_params.pkl`` does not exist in the
            given directory.
    """
    params_path = os.path.join(checkpoint_dir, "best_params.pkl")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"best_params.pkl not found: {params_path}")

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    # Infer model type and encoding from directory name
    dir_name = os.path.basename(checkpoint_dir)
    parts = dir_name.split("_")

    model_type = "conv"
    encoding_type = "onehot"
    biopart_type = "promoter"
    task_type = "regression"

    for part in parts:
        if part in ("conv", "1dcnn"):
            model_type = part
        if part in ("onehot", "single", "pairs", "triples"):
            encoding_type = part
        if part in ("promoter", "rbs", "terminator"):
            biopart_type = part
        if part in ("regression", "classification"):
            task_type = part

    seq_len_map = {"promoter": 40, "rbs": 15, "terminator": 50}
    seq_len = seq_len_map.get(biopart_type, 40)

    config: Dict[str, Any] = {
        "model_type": model_type,
        "encoding_type": encoding_type,
        "biopart_type": biopart_type,
        "seq_len": seq_len,
        "task_type": task_type,
        **params,
    }

    return config


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: str = "cuda:0",
) -> Tuple[torch.nn.Module, Optional[Any]]:
    """Load a DeepBioParts model from a checkpoint file.

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint file.
        config: Model configuration dictionary (as returned by
            :func:`load_model_config`).
        device: Torch device string.

    Returns:
        A ``(model, label_transform)`` tuple where *label_transform* is
        ``None`` when no transform was applied during training.
    """
    logger.info("Loading model from: %s", checkpoint_path)

    model = ModelFactory.create(
        config["model_type"],
        seq_len=config["seq_len"],
        num_classes=1,
        dropout_rate=config.get("dropout_rate", 0.2),
        conv_width_motif=config.get("conv_width_motif", 5),
        n_heads=config.get("n_heads", 16),
        conv_hidden=config.get("conv_hidden", 128),
        motif_conv_hidden=config.get("motif_conv_hidden", 256),
        vocab_size=None,  # one-hot encoding
        task_type="regression",
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        label_transform = checkpoint.get("label_transform", None)
    else:
        state_dict = checkpoint
        label_transform = None

    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Model loaded: %s", config["model_type"])
    logger.info("Sequence length: %s", config["seq_len"])

    return model, label_transform


# ======================================================================
# Importance extraction
# ======================================================================

def get_attention_weights_from_model(
    model: torch.nn.Module,
    sequences: List[str],
    seq_len: int,
    device: str = "cuda:0",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract attention-based importance weights from the model.

    For models with an ``attention`` attribute (e.g. CNN–Attention–BiLSTM), a forward
    hook captures the attention output and computes per-position importance
    as the L2 norm along the hidden dimension, softmax-normalised across
    positions.  When no attention layer is found, uniform weights are
    returned as a fallback.

    Args:
        model: A loaded, eval-mode model.
        sequences: List of DNA sequences.
        seq_len: Fixed sequence length expected by the model.
        device: Torch device string.
        batch_size: Mini-batch size for inference.

    Returns:
        A ``(predictions, attention_weights)`` tuple where
        *predictions* has shape ``(n_samples,)`` and *attention_weights*
        has shape ``(n_samples, seq_len)``.
    """
    model.eval()

    features = seq2onehot(sequences)
    features = torch.tensor(features, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(features, torch.zeros(len(features)))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    all_predictions: List[np.ndarray] = []
    attention_hook: List[torch.Tensor] = []

    def _hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:
        attention_output = output[0] if isinstance(output, tuple) else output
        importance = torch.norm(attention_output, dim=-1)  # [batch, seq_len]
        importance = torch.softmax(importance, dim=1)
        attention_hook.append(importance.detach().cpu())

    handle: Optional[Any] = None
    if hasattr(model, "attention"):
        handle = model.attention.register_forward_hook(_hook_fn)

    with torch.no_grad():
        for batch_features, _ in tqdm(dataloader, desc="Extracting attention"):
            batch_features = batch_features.permute(0, 2, 1).to(device)
            preds = model(batch_features)
            all_predictions.extend(preds.cpu().numpy())

    if handle is not None:
        handle.remove()

    if attention_hook:
        all_attention_weights = torch.cat(attention_hook, dim=0).numpy()
    else:
        seq_len_actual = features.shape[1]
        all_attention_weights = np.ones((len(features), seq_len_actual)) / seq_len_actual

    return np.array(all_predictions), all_attention_weights


def compute_gradient_based_importance(
    model: torch.nn.Module,
    sequences: List[str],
    seq_len: int,
    device: str = "cuda:0",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute position importance using gradient-based saliency.

    For each mini-batch the gradient of the sum of outputs with respect to
    the one-hot input is computed.  The maximum absolute gradient across the
    four nucleotide channels at each position gives a saliency score,
    which is then softmax-normalised across positions.

    Args:
        model: A loaded, eval-mode model.
        sequences: List of DNA sequences.
        seq_len: Fixed sequence length expected by the model.
        device: Torch device string.
        batch_size: Mini-batch size.

    Returns:
        A ``(predictions, importance_weights)`` tuple with shapes
        ``(n_samples,)`` and ``(n_samples, seq_len)`` respectively.
    """
    model.eval()

    features = seq2onehot(sequences)
    features = torch.tensor(features, dtype=torch.float32)

    all_predictions: List[np.ndarray] = []
    all_importance: List[np.ndarray] = []

    for i in tqdm(range(0, len(features), batch_size), desc="Computing gradient importance"):
        end_idx = min(i + batch_size, len(features))
        batch_features = features[i:end_idx]

        batch_features = batch_features.permute(0, 2, 1).to(device)
        batch_features.requires_grad_(True)

        outputs = model(batch_features)
        target = outputs.sum()
        target.backward()

        grads = batch_features.grad  # [batch, 4, seq_len]
        importance = torch.max(torch.abs(grads), dim=1)[0]  # [batch, seq_len]
        importance = torch.softmax(importance, dim=1)

        all_importance.append(importance.detach().cpu().numpy())
        all_predictions.extend(outputs.detach().cpu().numpy())

        batch_features.grad = None

    all_importance_arr = np.vstack(all_importance)
    return np.array(all_predictions), all_importance_arr


# ======================================================================
# Position-level statistics
# ======================================================================

def compute_position_importance(
    sequences: List[str],
    attention_weights: np.ndarray,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, Dict[str, Any]]]]:
    """Aggregate per-position importance statistics.

    Args:
        sequences: List of DNA sequences.
        attention_weights: Array of shape ``(n_samples, seq_len)`` with
            importance weights for each sample/position pair.

    Returns:
        A ``(position_scores, nucleotide_scores)`` tuple.

        *position_scores* maps each position index to a dict with keys
        ``mean``, ``std``, ``median``, ``max``.

        *nucleotide_scores* maps each position to a nested dict
        ``{nucleotide: {"mean": float, "count": int}}`` for nucleotides
        ``A``, ``T``, ``C``, ``G``.
    """
    max_len = attention_weights.shape[1]

    position_scores: Dict[int, Dict[str, float]] = {}
    for pos in range(max_len):
        scores_at_pos = attention_weights[:, pos]
        position_scores[pos] = {
            "mean": float(np.mean(scores_at_pos)),
            "std": float(np.std(scores_at_pos)),
            "median": float(np.median(scores_at_pos)),
            "max": float(np.max(scores_at_pos)),
        }

    nucleotide_scores: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for pos in range(max_len):
        nucleotide_scores[pos] = {}
        for nuc in ("A", "T", "C", "G"):
            mask = np.array(
                [
                    (pos < len(seq)) and (seq[pos] == nuc)
                    for seq in sequences
                ]
            )
            if mask.sum() > 0:
                nucleotide_scores[pos][nuc] = {
                    "mean": float(np.mean(attention_weights[mask, pos])),
                    "count": int(mask.sum()),
                }
            else:
                nucleotide_scores[pos][nuc] = {"mean": 0.0, "count": 0}

    return position_scores, nucleotide_scores


# ======================================================================
# Report persistence
# ======================================================================

def save_position_importance_report(
    sequences: List[str],
    attention_weights: np.ndarray,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Save position importance scores as CSV files.

    Three files are written to *output_dir*:

    * ``position_importance.csv`` -- per-position summary statistics.
    * ``nucleotide_position_importance.csv`` -- nucleotide-specific scores.
    * ``importance_weights.npy`` -- raw importance weight array.

    Args:
        sequences: List of DNA sequences.
        attention_weights: Array of shape ``(n_samples, seq_len)``.
        output_dir: Directory where files are written.

    Returns:
        A ``(position_df, nucleotide_df)`` tuple of the DataFrames that
        were saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    position_scores, nucleotide_scores = compute_position_importance(
        sequences, attention_weights
    )

    # Position-level scores
    pos_df = pd.DataFrame(position_scores).T
    pos_df.index.name = "position"
    pos_path = os.path.join(output_dir, "position_importance.csv")
    pos_df.to_csv(pos_path)
    logger.info("Saved position importance to: %s", pos_path)

    # Nucleotide-specific scores
    nuc_data: List[Dict[str, Any]] = []
    for pos in nucleotide_scores:
        for nuc in ("A", "T", "C", "G"):
            nuc_data.append(
                {
                    "position": pos,
                    "nucleotide": nuc,
                    "mean_importance": nucleotide_scores[pos][nuc]["mean"],
                    "count": nucleotide_scores[pos][nuc]["count"],
                }
            )

    nuc_df = pd.DataFrame(nuc_data)
    nuc_path = os.path.join(output_dir, "nucleotide_position_importance.csv")
    nuc_df.to_csv(nuc_path, index=False)
    logger.info("Saved nucleotide position importance to: %s", nuc_path)

    # Raw weights
    weights_path = os.path.join(output_dir, "importance_weights.npy")
    np.save(weights_path, attention_weights)
    logger.info("Saved importance weights to: %s", weights_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("Position Importance Summary")
    logger.info("=" * 60)
    logger.info("Total samples: %d", len(sequences))
    logger.info("Sequence length: %d", attention_weights.shape[1])
    logger.info("Top 10 most important positions (by mean importance):")
    top_positions = sorted(
        position_scores.items(), key=lambda x: x[1]["mean"], reverse=True
    )[:10]
    for pos, scores in top_positions:
        logger.info(
            "  Position %3d: mean=%.6f, std=%.6f, median=%.6f, max=%.6f",
            pos,
            scores["mean"],
            scores["std"],
            scores["median"],
            scores["max"],
        )
    logger.info("=" * 60)

    return pos_df, nuc_df


# ======================================================================
# Full evaluation pipeline
# ======================================================================

def evaluate_position_importance(
    checkpoint_dir: str,
    data_path: str,
    output_dir: str,
    device: str = "cuda:0",
    batch_size: int = 32,
    fold_id: Optional[int] = None,
    use_gradient: bool = False,
) -> Dict[str, Any]:
    """Run the position-importance evaluation pipeline.

    The pipeline loads a model checkpoint, computes per-position importance
    weights (via attention or gradient saliency), and saves CSV reports.
    All results are also returned in a structured dictionary so callers
    can inspect or further process them without touching the filesystem.

    Args:
        checkpoint_dir: Path to a checkpoint directory (contains
            ``fold_1/`` ... ``fold_5/`` sub-directories).
        data_path: Path to a CSV file whose first column holds DNA
            sequences and whose optional second column holds activity
            labels.
        output_dir: Directory where CSV reports are written.
        device: Torch device string.
        batch_size: Mini-batch size for inference.
        fold_id: Specific fold to use (1-5).  ``None`` uses the first
            available fold.
        use_gradient: If ``True`` use gradient saliency instead of
            attention weights.

    Returns:
        A dictionary with the following keys:

        * ``config`` -- model configuration dict
        * ``sequences`` -- list of evaluated sequences
        * ``actuals`` -- numpy array of ground-truth labels (or ``None``)
        * ``predictions`` -- numpy array of model predictions
        * ``importance_weights`` -- numpy array of shape
          ``(n_samples, seq_len)``
        * ``position_scores`` -- per-position statistics dict
        * ``nucleotide_scores`` -- per-position, per-nucleotide dict
        * ``position_df`` / ``nucleotide_df`` -- saved DataFrames
        * ``label_transform`` -- the label transform info (or ``None``)
    """
    logger.info("=" * 60)
    logger.info("Position Importance Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", checkpoint_dir)
    logger.info("Data: %s", data_path)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    # ---- Configuration ----
    config = load_model_config(checkpoint_dir)
    logger.info("Model configuration:")
    for key, value in config.items():
        logger.info("  %s: %s", key, value)

    # ---- Data ----
    logger.info("Loading evaluation data from: %s", data_path)
    df = pd.read_csv(data_path)
    sequences: List[str] = df.iloc[:, 0].tolist()

    actuals: Optional[np.ndarray] = None
    if df.shape[1] >= 2:
        actuals = df.iloc[:, 1].values.astype(np.float32)

    logger.info("Total samples: %d", len(sequences))
    if actuals is not None:
        logger.info("Activity range: [%.4f, %.4f]", actuals.min(), actuals.max())

    # ---- Locate checkpoint ----
    if fold_id is not None:
        fold_paths = [os.path.join(checkpoint_dir, f"fold_{fold_id}", "checkpoint.pth")]
    else:
        fold_paths = [
            os.path.join(checkpoint_dir, f"fold_{i}", "checkpoint.pth")
            for i in range(1, 6)
        ]

    checkpoint_path: Optional[str] = None
    for path in fold_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    # ---- Load model ----
    model, label_transform = load_model_from_checkpoint(checkpoint_path, config, device)

    # ---- Extract importance ----
    logger.info("Extracting position importance weights...")
    if use_gradient:
        predictions, importance_weights = compute_gradient_based_importance(
            model, sequences, config["seq_len"], device, batch_size
        )
    else:
        predictions, importance_weights = get_attention_weights_from_model(
            model, sequences, config["seq_len"], device, batch_size
        )

    # Inverse-transform predictions if log10 was used
    if label_transform == 'log10':
        predictions = np.power(10.0, predictions) - 1.0

    # ---- Save reports ----
    logger.info("Saving position importance reports...")
    os.makedirs(output_dir, exist_ok=True)
    position_df, nucleotide_df = save_position_importance_report(
        sequences, importance_weights, output_dir
    )

    # ---- Compute structured stats ----
    position_scores, nucleotide_scores = compute_position_importance(
        sequences, importance_weights
    )

    logger.info("=" * 60)
    logger.info("Evaluation complete! Results saved to: %s", output_dir)
    logger.info("=" * 60)

    return {
        "config": config,
        "sequences": sequences,
        "actuals": actuals,
        "predictions": predictions,
        "importance_weights": importance_weights,
        "position_scores": position_scores,
        "nucleotide_scores": nucleotide_scores,
        "position_df": position_df,
        "nucleotide_df": nucleotide_df,
        "label_transform": label_transform,
    }
