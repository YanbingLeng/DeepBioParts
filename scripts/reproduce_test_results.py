#!/usr/bin/env python3
"""Rerun DeepBioParts inference on the four released test datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
EXPECTED_ROWS = 100
STRONG_TERMINATOR_THRESHOLD = 95.0

TASKS = (
    {
        "name": "promoter",
        "biopart": "promoter",
        "data_file": "promoter_test_100.csv",
        "label_col": "true_activity",
        "sequence_length": 40,
        "task_type": "regression",
        "model_dir": "predictor_checkpoints/language_model/promoter_LoRA_finetune",
        "summary_metric": "Pearson_r",
    },
    {
        "name": "rbs",
        "biopart": "rbs",
        "data_file": "rbs_test_100.csv",
        "label_col": "true_activity",
        "sequence_length": 15,
        "task_type": "regression",
        "model_dir": "predictor_checkpoints/supervised_model/rbs_CNN_Attn_BiLSTM",
        "summary_metric": "Pearson_r",
    },
    {
        "name": "terminator_continuing",
        "biopart": "terminator",
        "data_file": "terminator_continuing_test_100.csv",
        "label_col": "true_activity",
        "sequence_length": 50,
        "task_type": "regression",
        "model_dir": "predictor_checkpoints/supervised_model/terminator_CNN_Attn_BiLSTM",
        "summary_metric": "Pearson_r",
    },
    {
        "name": "terminator_strong",
        "biopart": "terminator",
        "data_file": "terminator_strong_test_100.csv",
        "label_col": "activity",
        "sequence_length": 50,
        "task_type": "classification",
        "model_dir": "predictor_checkpoints/supervised_model/terminator_CNN_Attn_BiLSTM_cla",
        "summary_metric": "AUC",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the released model weights, predict all sequences in the four independent "
            "test sets, and recompute the reported Pearson correlations and AUROC."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/test"),
        help="Directory containing the four test-set CSV files (default: data/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/test_reproduction"),
        help="Directory for new predictions and metrics (default: results/test_reproduction).",
    )
    parser.add_argument("--device", default="cuda:0", help="PyTorch device (default: cuda:0).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size; 8 is a conservative default for the Evo promoter model.",
    )
    return parser.parse_args()


def resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_test_frame(frame, path: Path, task: dict) -> None:
    expected_columns = ("part_id", "sequence", task["label_col"])
    if tuple(frame.columns) != expected_columns:
        raise ValueError(
            f"{path} must contain exactly {list(expected_columns)}; found {list(frame.columns)}"
        )
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"{path} contains {len(frame)} rows; expected {EXPECTED_ROWS}")
    if frame[list(expected_columns)].isna().any().any():
        raise ValueError(f"{path} contains missing values")
    if frame["part_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate part_id values")

    sequences = frame["sequence"].astype(str).str.upper()
    if not sequences.str.fullmatch("[ACGT]+", na=False).all():
        raise ValueError(f"{path} contains a non-ACGT sequence")
    if not sequences.str.len().eq(task["sequence_length"]).all():
        raise ValueError(
            f"{path} contains a sequence with an unexpected length; "
            f"expected {task['sequence_length']} bp"
        )


def require_runtime_files(tasks: tuple[dict, ...]) -> None:
    missing = []
    for task in tasks:
        model_dir = PROJECT_ROOT / task["model_dir"]
        if not model_dir.is_dir():
            missing.append(str(model_dir))

    evo_backbone = PROJECT_ROOT / "src/evo/models/evo-1.5-8k-base"
    if not evo_backbone.is_dir():
        missing.append(str(evo_backbone))

    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Required model files are missing. Follow README sections 2 and 3, then retry:\n"
            f"  - {formatted}"
        )


def run_task(task: dict, data_dir: Path, output_dir: Path, device: str, batch_size: int):
    import pandas as pd

    from src.evaluation.predictor_inference import (
        compute_classification_metrics,
        compute_regression_metrics,
        ensemble_predict,
        find_fold_checkpoints,
        load_label_transform,
    )

    data_path = data_dir / task["data_file"]
    if not data_path.is_file():
        raise FileNotFoundError(f"Test-set file not found: {data_path}")
    frame = pd.read_csv(data_path)
    validate_test_frame(frame, data_path, task)

    model_dir = PROJECT_ROOT / task["model_dir"]
    checkpoints = find_fold_checkpoints(model_dir)
    label_transform = load_label_transform(model_dir)
    if task["name"] == "terminator_continuing" and label_transform is None:
        label_transform = {"transform": "log10", "shift": 1.0}

    prediction = ensemble_predict(
        checkpoints=checkpoints,
        sequences=frame["sequence"].astype(str).tolist(),
        biopart=task["biopart"],
        device=device,
        batch_size=batch_size,
        label_transform=label_transform,
    )
    if prediction["task_type"] != task["task_type"]:
        raise ValueError(
            f"{task['name']} expected a {task['task_type']} model, "
            f"but the checkpoint was detected as {prediction['task_type']}"
        )

    task_output = output_dir / task["name"]
    task_output.mkdir(parents=True, exist_ok=True)
    results = frame.copy()

    if task["task_type"] == "regression":
        measured = frame[task["label_col"]].to_numpy(dtype=float)
        predicted = prediction["mean_pred"]
        metrics = compute_regression_metrics(measured, predicted)
        results["predicted_activity"] = predicted
        results["prediction_std"] = prediction["std_pred"]
        results["residual"] = measured - predicted
    else:
        activity = frame[task["label_col"]].to_numpy(dtype=float)
        labels = (activity > STRONG_TERMINATOR_THRESHOLD).astype(int)
        probabilities = prediction["mean_probs"]
        predicted_classes = prediction["predicted_classes"]
        metrics = compute_classification_metrics(labels, predicted_classes, probabilities)
        results["true_class"] = labels
        results["predicted_class"] = predicted_classes
        results["prob_class_0"] = probabilities[:, 0]
        results["prob_class_1"] = probabilities[:, 1]
        results["prediction_std"] = prediction["std_logits"].mean(axis=1)

    results.to_csv(task_output / "predictions.csv", index=False, float_format="%.8f")
    pd.DataFrame([metrics]).to_csv(task_output / "metrics.csv", index=False)
    return {
        "dataset": task["name"],
        "task": task["task_type"],
        "n": len(frame),
        "metric": task["summary_metric"],
        "value": float(metrics[task["summary_metric"]]),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    data_dir = resolve_from_root(args.data_dir)
    output_dir = resolve_from_root(args.output_dir)
    require_runtime_files(TASKS)

    print("DeepBioParts end-to-end test-set reproduction")
    print(f"Device: {args.device}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")

    summary = []
    for task in TASKS:
        print(f"\n{'=' * 72}\nRunning {task['name']}\n{'=' * 72}")
        summary.append(
            run_task(task, data_dir, output_dir, args.device, args.batch_size)
        )

    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(output_dir / "summary.csv", index=False, float_format="%.8f")
    print("\nReproduced test-set metrics")
    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
