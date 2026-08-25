"""Shared predictor-inference library.

Centralizes predictor inference logic (5-fold ensemble, automatic model-type
detection, sliding window, activity transform, metrics) so it can be shared by
``scripts/predict.py``, ``scripts/fig3/plot_fig3bcd_predictor_eval.py``, and
others without re-implementation. This module performs only computation and
returns structured data (dict / DataFrame); CLI parsing, output formatting,
and plotting remain in the caller scripts.

Behavior contract: these functions are ported verbatim from the original inline
code in predict.py / plot_fig3bcd and are behavior-preserving (checkpoints are
loaded as state_dicts, independent of class paths).

Evo-related imports are kept lazy (inside function bodies), so the py39
environment can import this module without Evo dependencies; the heavy Evo
dependencies are only triggered when a checkpoint is actually an Evo model.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.config import SEQ_LENGTHS, load_config
from src.utils.data import seq2onehot

# Project root (src/evaluation/predictor_inference.py -> up two levels)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Path / checkpoint discovery
# ---------------------------------------------------------------------------

def detect_biopart_from_path(model_dir: Path) -> str:
    """Infer the biopart type from the model directory path."""
    full_path = str(model_dir).lower()
    if "rbs" in full_path:
        return "rbs"
    elif "promoter" in full_path:
        return "promoter"
    elif "terminator" in full_path:
        return "terminator"
    else:
        raise ValueError(
            f"Cannot infer biopart type from path '{full_path}'. "
            f"Please specify it explicitly with --biopart."
        )


def load_label_transform(model_dir: Path) -> Optional[dict]:
    """Load label-transform parameters."""
    transform_path = model_dir / "label_transform.json"
    if transform_path.exists():
        with open(transform_path, "r") as f:
            return json.load(f)
    return None


def find_fold_checkpoints(model_dir: Path) -> list:
    """Find checkpoint files for all folds."""
    checkpoints = []
    for i in range(1, 6):
        fold_dir = model_dir / f"fold_{i}"
        if not fold_dir.exists():
            continue
        for ckpt_name in ["checkpoint.pth", "checkpoint.pt", "best.pt", "best.pth"]:
            ckpt_path = fold_dir / ckpt_name
            if ckpt_path.exists():
                checkpoints.append(ckpt_path)
                break
        else:
            pt_files = list(fold_dir.glob("*.pt")) + list(fold_dir.glob("*.pth"))
            if pt_files:
                checkpoints.append(pt_files[0])

    if len(checkpoints) == 0:
        for ckpt_name in ["best_model.pth", "best_model.pt", "best.pth", "best.pt",
                           "checkpoint.pth", "checkpoint.pt", "model.pth", "model.pt"]:
            ckpt_path = model_dir / ckpt_name
            if ckpt_path.exists():
                checkpoints.append(ckpt_path)
                break
        else:
            pt_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
            if pt_files:
                checkpoints.append(pt_files[0])

    if len(checkpoints) == 0:
        raise FileNotFoundError(f"No checkpoint file found in {model_dir}")
    if len(checkpoints) == 1:
        print(f"Using a single model file: {checkpoints[0].name}")
    elif len(checkpoints) < 5:
        print(f"Warning: only {len(checkpoints)} fold models found (expected 5)")

    return sorted(checkpoints)


# ---------------------------------------------------------------------------
# Model loading (Evo + traditional DL)
# ---------------------------------------------------------------------------

def infer_model_hyperparams(state_dict: dict) -> dict:
    """Infer model hyperparameters from a state_dict."""
    params = {}
    if 'conv1.weight' in state_dict:
        params['motif_conv_hidden'] = state_dict['conv1.weight'].shape[0]
        params['conv_width_motif'] = state_dict['conv1.weight'].shape[2]
    if 'conv2.weight' in state_dict:
        params['conv_hidden'] = state_dict['conv2.weight'].shape[0]
    if 'attention.self_attn.out_proj.weight' in state_dict:
        embed_dim = state_dict['attention.self_attn.out_proj.weight'].shape[0]
        if 'conv_hidden' in params:
            if embed_dim % params['conv_hidden'] == 0:
                params['n_heads'] = embed_dim // params['conv_hidden']
    return params


def detect_task_type(state_dict: dict, checkpoint_path: Path) -> tuple:
    """Detect the model task type and number of classes."""
    has_ornet = any('ornet' in k for k in state_dict.keys())
    if has_ornet:
        for k, v in state_dict.items():
            if 'ornet.or_bias' in k:
                return 'classification', v.shape[0] + 1
        return 'classification', 2

    has_evo_cla_head = any('head.classification' in k for k in state_dict.keys())
    if has_evo_cla_head:
        min_out_dim = None
        for k, v in state_dict.items():
            if 'head.classification' in k and v.ndim == 2:
                if min_out_dim is None or v.shape[0] < min_out_dim:
                    min_out_dim = v.shape[0]
        if min_out_dim is not None and min_out_dim >= 2:
            return 'classification', min_out_dim
        return 'classification', 2

    path_str = str(checkpoint_path).lower()
    if 'classification' in path_str or '_cla_' in path_str:
        return 'classification', 2

    return 'regression', 1


def load_predictor(checkpoint_path: Path, biopart: str, device: str,
                   task_type: str = 'regression', num_classes: int = 1):
    """Load a single predictor model (supports Evo and traditional DL)."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    is_evo_model = any(k.startswith('evo.') for k in state_dict.keys())

    if is_evo_model:
        print("    Detected Evo model; using EvoWithRegressionHead...")
        evo_dir = PROJECT_ROOT / "src" / "evo"
        if str(evo_dir) not in sys.path:
            sys.path.insert(0, str(evo_dir))
        from src.evo.lora_finetune_evo import EvoWithRegressionHead, load_evo_from_local

        local_model_path = str(evo_dir / "models" / "evo-1.5-8k-base")
        evo_base = load_evo_from_local(
            local_model_path=local_model_path,
            model_name='evo-1.5-8k-base',
            device=device,
        )

        lora_keys = [k for k in state_dict.keys() if 'lora_A' in k]
        lora_rank = state_dict[lora_keys[0]].shape[0] if lora_keys else 16

        # Auto-restore ablation configuration; older checkpoints without the
        # key fall back to lora/attention (backward compatible)
        ck_evo_adaptation = checkpoint.get('evo_adaptation') if isinstance(checkpoint, dict) else None
        ck_pooling_mode = checkpoint.get('pooling_mode', 'attention') if isinstance(checkpoint, dict) else 'attention'

        model = EvoWithRegressionHead(
            evo_base, hidden_dim=512, dropout=0.2,
            use_lora=True, lora_r=lora_rank, lora_alpha=32, lora_dropout=0.1,
            task_type=task_type,
            evo_adaptation=ck_evo_adaptation,
            pooling_mode=ck_pooling_mode,
        ).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    else:
        print("    Detected traditional DL model; using ModelFactory...")
        # Prefer the facade (src/models/__init__.py, which internally imports
        # 'models.predictor') so the same predictor.py is not loaded twice
        # under two names ('src.models.predictor' vs 'models.predictor'),
        # which would re-trigger @ModelFactory.register printing. The facade
        # requires SRC_DIR on sys.path; when the calling script only adds
        # PROJECT_ROOT it silently degrades to None, in which case fall back
        # to importing the src.models.predictor submodule directly (which
        # needs only PROJECT_ROOT).
        from src.models import ModelFactory
        if ModelFactory is None:
            from src.models.predictor import ModelFactory

        # Determine the architecture from weight keys (directory names are
        # unreliable: conv-family directories are named e.g. CNN_Attn_BiLSTM,
        # which contains neither "conv" nor distinguishes _CNN from
        # _CNN_BiLSTM). ConvPredictor has bilstm/norm1 modules; OneDCNN has
        # bn1/fc modules.
        if any(k.startswith(("bilstm", "norm1")) for k in state_dict.keys()):
            model_type = "conv"
        elif any(k.startswith(("bn1", "fc")) for k in state_dict.keys()):
            model_type = "1dcnn"
        else:
            model_type = "conv"

        inferred_params = infer_model_hyperparams(state_dict)
        print(f"    Inferred hyperparameters: {inferred_params}")

        # Auto-restore ablation configuration; older checkpoints (raw
        # state_dict) without the key fall back to 'full'
        ablation_variant = checkpoint.get('ablation_variant', 'full') if isinstance(checkpoint, dict) else 'full'

        seq_len = SEQ_LENGTHS.get(biopart, 50)
        model = ModelFactory.create(
            model_type, seq_len=seq_len, num_classes=num_classes,
            dropout_rate=inferred_params.get('dropout_rate', 0.2),
            conv_width_motif=inferred_params.get('conv_width_motif', 5),
            n_heads=inferred_params.get('n_heads', 16),
            conv_hidden=inferred_params.get('conv_hidden', 128),
            motif_conv_hidden=inferred_params.get('motif_conv_hidden', 256),
            vocab_size=None, task_type=task_type,
            ablation_variant=ablation_variant,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Prediction core
# ---------------------------------------------------------------------------

def inverse_log_label(predictions: np.ndarray, shift: float = 1.0) -> np.ndarray:
    """Inverse of log10(label+shift): 10^pred - shift."""
    return np.power(10.0, predictions) - shift


def ornet_logits_to_probs(logits: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert ORNet ordinal logits into class probabilities."""
    sig = 1.0 / (1.0 + np.exp(-logits))
    n = sig.shape[0]
    probs = np.zeros((n, num_classes), dtype=np.float64)
    probs[:, 0] = 1.0 - sig[:, 0]
    for k in range(1, num_classes - 1):
        probs[:, k] = sig[:, k - 1] - sig[:, k]
    probs[:, num_classes - 1] = sig[:, num_classes - 2]
    return probs


def ensemble_predict(
    checkpoints: list,
    sequences: list,
    biopart: str,
    device: str,
    batch_size: int = 256,
    label_transform: Optional[dict] = None,
) -> dict:
    """Ensemble prediction (5-fold model averaging); supports regression and classification."""
    print(f"\nLoading {len(checkpoints)} models for ensemble prediction...")

    first_checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
    first_state_dict = first_checkpoint.get('model_state_dict', first_checkpoint)
    is_evo_model = any(k.startswith('evo.') for k in first_state_dict.keys())

    task_type, num_classes = detect_task_type(first_state_dict, checkpoints[0])
    print(f"Detected task type: {task_type}" +
          (f", num_classes: {num_classes}" if task_type == 'classification' else ''))

    if is_evo_model:
        print("Detected Evo model; using token encoding...")
        evo_dir = PROJECT_ROOT / "src" / "evo"
        if str(evo_dir) not in sys.path:
            sys.path.insert(0, str(evo_dir))
        from src.evo.lora_finetune_evo import dna_sequence_to_tokens

        seq_len = SEQ_LENGTHS.get(biopart, 50)
        tokens_list = [dna_sequence_to_tokens(seq, seq_len) for seq in sequences]
        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
    else:
        print("Detected traditional DL model; using one-hot encoding...")
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        onehot = seq2onehot(sequences)

    evo_model = None
    all_predictions = []

    for i, ckpt_path in enumerate(checkpoints, 1):
        print(f"  [{i}/{len(checkpoints)}] Loading model: {ckpt_path.name}")
        try:
            if is_evo_model:
                if evo_model is None:
                    model = load_predictor(ckpt_path, biopart, device,
                                           task_type=task_type, num_classes=num_classes)
                    evo_model = model
                else:
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    fold_sd = ckpt.get('model_state_dict', ckpt)
                    evo_model.load_state_dict(fold_sd, strict=False)
                    evo_model.eval()
                    model = evo_model
            else:
                model = load_predictor(ckpt_path, biopart, device,
                                       task_type=task_type, num_classes=num_classes)

            if is_evo_model:
                ds = torch.utils.data.TensorDataset(tokens_tensor)
                loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
                fold_preds = []
                with torch.no_grad():
                    for batch in loader:
                        outputs = model(batch[0].to(device))
                        if task_type == 'classification':
                            fold_preds.append(outputs.detach().cpu().numpy())
                        else:
                            fold_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_predictions.append(
                    np.concatenate(fold_preds, axis=0) if task_type == 'classification'
                    else np.array(fold_preds))
            else:
                ds = torch.utils.data.TensorDataset(
                    torch.tensor(onehot, dtype=torch.float32))
                loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
                fold_preds = []
                with torch.no_grad():
                    for batch in loader:
                        batch = batch[0].permute(0, 2, 1).to(device)  # (B, 4, L)
                        outputs = model(batch)
                        if task_type == 'classification':
                            fold_preds.append(outputs.detach().cpu().numpy())
                        else:
                            fold_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_predictions.append(
                    np.concatenate(fold_preds, axis=0) if task_type == 'classification'
                    else np.array(fold_preds))

            if not is_evo_model:
                del model
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Warning: failed to load or predict with model {ckpt_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if evo_model is not None:
        del evo_model
        torch.cuda.empty_cache()

    if len(all_predictions) == 0:
        raise RuntimeError("No models were loaded successfully")

    print(f"\nEnsemble prediction complete using {len(all_predictions)} models")

    if task_type == 'classification':
        predictions_array = np.array(all_predictions)
        mean_logits = predictions_array.mean(axis=0)
        std_logits = predictions_array.std(axis=0)
        if mean_logits.ndim == 1:
            mean_logits = mean_logits[:, np.newaxis]
            std_logits = std_logits[:, np.newaxis]
        if is_evo_model:
            from scipy.special import softmax as scipy_softmax
            mean_probs = scipy_softmax(mean_logits, axis=-1)
        else:
            mean_probs = ornet_logits_to_probs(mean_logits, num_classes)
        predicted_classes = np.argmax(mean_probs, axis=1)
        return {
            'task_type': 'classification',
            'mean_logits': mean_logits,
            'std_logits': std_logits,
            'mean_probs': mean_probs,
            'predicted_classes': predicted_classes,
            'num_classes': num_classes,
        }
    else:
        predictions_array = np.array(all_predictions)
        mean_pred = predictions_array.mean(axis=0)
        std_pred = predictions_array.std(axis=0)
        if label_transform is not None and label_transform.get('transform') == 'log10':
            print("Applying log10 inverse transform...")
            shift = label_transform.get('shift', 1.0)
            mean_pred = inverse_log_label(mean_pred, shift)
            std_pred = std_pred * np.log(10) * np.power(10.0, predictions_array.mean(axis=0))
        return {
            'task_type': 'regression',
            'mean_pred': mean_pred,
            'std_pred': std_pred,
        }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression evaluation metrics."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import spearmanr

    metrics = {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Pearson_r": np.corrcoef(y_true, y_pred)[0, 1],
    }
    spearman_corr, _ = spearmanr(y_true, y_pred)
    metrics["Spearman_r"] = spearman_corr
    return metrics


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_probs: np.ndarray) -> dict:
    """Compute classification evaluation metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    if len(np.unique(y_true)) == 2 and y_probs.shape[1] == 2:
        from sklearn.metrics import roc_auc_score
        try:
            metrics["AUC"] = roc_auc_score(y_true.astype(int), y_probs[:, 1])
        except ValueError:
            pass
    return metrics


# ---------------------------------------------------------------------------
# Activity transform
# ---------------------------------------------------------------------------

def load_activity_transform(biopart: str) -> Optional[dict]:
    """Load the activity-transform parameters from the config file."""
    try:
        config = load_config(biopart)
        transform = config.get("activity_transform", {})
        if "slope" in transform:
            slope = float(transform["slope"])
            intercept = float(transform.get("intercept", 0.0))
            print(f"  Activity transform: {transform.get('formula', f'{slope} * pred + {intercept}')}")
            return {"slope": slope, "intercept": intercept}
    except Exception as e:
        print(f"  Warning: failed to load activity transform ({e})")
    return None


def apply_activity_transform(predictions: np.ndarray, transform: Optional[dict]) -> np.ndarray:
    """Apply the activity transform: true_activity = slope * pred + intercept."""
    if transform is None:
        return predictions
    return transform.get("slope", 1.0) * predictions + transform.get("intercept", 0.0)


def get_default_model_dir(biopart: str) -> str:
    """Get the default predictor model path from the config file."""
    config = load_config(biopart)
    if "default_predictor" in config:
        return config["default_predictor"]
    predictors = config.get("predictors", {})
    for model_key in ["dl_model", "conv_model", "evo_model"]:
        model_cfg = predictors.get(model_key, {})
        if model_cfg.get("enabled", False):
            path = model_cfg.get("predictor_dir") or model_cfg.get("checkpoint_path")
            if path:
                return path
    raise ValueError(
        f"No default predictor path found for {biopart}. "
        f"Set default_predictor in configs/bioparts/{biopart}.yaml, "
        f"or specify --model_dir explicitly."
    )


def get_classification_model_dir(biopart: str) -> Optional[str]:
    """Get the classification predictor path from config, if configured.

    Used by joint scan to predict strong-part probability (e.g. the terminator
    ORNet classification model) instead of the regression default. Returns
    None when no ``classification_predictor`` is set for this biopart, so the
    caller falls back to the regression default.
    """
    config = load_config(biopart)
    return config.get("classification_predictor")


# ---------------------------------------------------------------------------
# Sliding-window prediction
# ---------------------------------------------------------------------------

def predict_with_sliding_window(
    sequence: str,
    biopart: str,
    checkpoints: list,
    device: str,
    batch_size: int = 256,
    scan_step: int = 1,
    label_transform: Optional[dict] = None,
    activity_transform: Optional[dict] = None,
) -> pd.DataFrame:
    """Predict the activity of a long sequence using a sliding window."""
    expected_len = SEQ_LENGTHS.get(biopart, 50)
    seq_len = len(sequence)

    if seq_len < expected_len:
        padded = sequence + 'N' * (expected_len - seq_len)
        windows = [(0, seq_len, padded)]
        print(f"  Sequence {seq_len}bp < expected {expected_len}bp; right-padded with 'N' to {expected_len}bp")
    elif seq_len == expected_len:
        windows = [(0, seq_len, sequence)]
    else:
        windows = []
        for i in range(0, seq_len - expected_len + 1, scan_step):
            windows.append((i, i + expected_len, sequence[i:i + expected_len]))
        print(f"  Sliding window: {seq_len}bp -> window {expected_len}bp, "
              f"step {scan_step}bp, {len(windows)} windows total")

    window_seqs = [w[2] for w in windows]
    result = ensemble_predict(
        checkpoints=checkpoints, sequences=window_seqs,
        biopart=biopart, device=device, batch_size=batch_size,
        label_transform=label_transform,
    )

    rows = []
    is_regression = result['task_type'] == 'regression'
    for i, (start, end, window_seq) in enumerate(windows):
        row = {'start': start, 'end': end, 'sequence': window_seq}
        if is_regression:
            raw_pred = result['mean_pred'][i]
            true_act = apply_activity_transform(np.array([raw_pred]), activity_transform)[0]
            row['predicted_fitness'] = float(raw_pred)
            row['predicted_activity'] = float(true_act)
            row['prediction_std'] = float(result['std_pred'][i])
        else:
            row['predicted_class'] = int(result['predicted_classes'][i])
            row['max_probability'] = float(result['mean_probs'][i].max())
            # Probability of the positive class (the last one); for binary
            # classification this is the probability of a "strong" element.
            row['prob_positive'] = float(result['mean_probs'][i][-1])
        rows.append(row)

    return pd.DataFrame(rows)
