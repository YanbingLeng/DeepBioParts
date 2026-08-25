"""Multi-part joint sliding-window scanning (inference orchestration, library layer).

``scan_all_bioparts``: runs the rbs / promoter / terminator models in turn on
a single unknown sequence, sliding each model over its expected window length,
and merges the results into one DataFrame. It only composes primitives from
``predictor_inference`` and performs no plotting.

Decoupled from the CLI; relative model paths are resolved against the project
root.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd
import torch

from src.config import SEQ_LENGTHS
from src.evaluation.predictor_inference import (
    find_fold_checkpoints,
    get_classification_model_dir,
    get_default_model_dir,
    load_activity_transform,
    load_label_transform,
    predict_with_sliding_window,
)

__all__ = ["ALL_BIOPARTS", "scan_all_bioparts"]

# Project root (src/evaluation/multi_part_scan.py -> parents[2])
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The three bioparts in the joint scan (promoter uses the Evo model, which
# requires an environment with flash-attn)
ALL_BIOPARTS = ["rbs", "promoter", "terminator"]


def scan_all_bioparts(
    sequence: str,
    device: str,
    batch_size: int = 256,
    scan_step: int = 1,
    no_transform: bool = False,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Run the sliding-window scan of the three biopart models on the same
    sequence and merge the results.

    Args:
        sequence: DNA sequence to scan (uppercase).
        device: inference device.
        batch_size: prediction batch size.
        scan_step: sliding step size (bp).
        no_transform: if True, do not apply the activity transform formula.
        allow_short: if True, pad sequences shorter than the window with N on
            the right and still predict.

    Returns:
        Merged DataFrame with columns including biopart, start, end, sequence,
        predicted_fitness, predicted_activity, prediction_std, etc.; an empty
        DataFrame if every scan fails.
    """
    seq_len = len(sequence)
    all_dfs = []

    for biopart in ALL_BIOPARTS:
        expected_len = SEQ_LENGTHS.get(biopart, 50)
        print(f"\n{'-'*50}")
        print(f"[{biopart}] scanning... (window {expected_len}bp, sequence {seq_len}bp)")

        if seq_len < expected_len and not allow_short:
            print(f"  Skipping: sequence {seq_len}bp < window {expected_len}bp")
            continue

        try:
            # Joint-scan model coverage: terminator uses the classification
            # model (P(strong)); the others use their default regression models
            override = get_classification_model_dir(biopart)
            if override:
                model_dir_str = override
                activity_transform = None
                print(f"  Joint scan mode: using classification model {override}")
            else:
                model_dir_str = get_default_model_dir(biopart)
                activity_transform = None if no_transform else load_activity_transform(biopart)

            model_dir = Path(model_dir_str)
            if not model_dir.is_absolute():
                model_dir = PROJECT_ROOT / model_dir
            model_dir = model_dir.resolve()

            if not model_dir.exists():
                print(f"  Skipping: model directory does not exist {model_dir}")
                continue

            checkpoints = find_fold_checkpoints(model_dir)
            label_transform = load_label_transform(model_dir)
        except Exception as e:
            print(f"  Skipping: failed to load model ({e})")
            continue

        try:
            window_df = predict_with_sliding_window(
                sequence, biopart, checkpoints, device,
                batch_size, scan_step,
                label_transform, activity_transform,
            )
            window_df.insert(0, "biopart", biopart)
            all_dfs.append(window_df)
            print(f"  Done: {len(window_df)} windows")
        except Exception as e:
            print(f"  Skipping: prediction failed ({e})")
            traceback.print_exc()
            continue

        torch.cuda.empty_cache()

    if not all_dfs:
        print("\nWarning: all biopart scans failed")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)
