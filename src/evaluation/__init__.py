"""Evaluation pipeline modules (library layer).

Provides pure computation functions for model and sequence evaluation.
All plotting logic is in ``visualization/`` — these modules return
structured data (dicts, arrays) suitable for downstream visualization.

.. note:: This is the canonical library layer for evaluation computations
   (the REFACTOR_PLAN target). Some figure scripts still inline evaluation
   logic and have not been fully migrated here; new code should prefer
   ``from evaluation import ...``.

Sub-modules:
    predictor_inference: Shared predictor loading + ensemble inference + sliding window + activity transform (used by predict.py / figure scripts).
    expression_comparison: Activity strength statistical comparison.
    nrp_analysis: Non-repetitive sequence analysis and Monte Carlo sampling.
    dataset_evaluation: Dataset quality scoring and subset extraction.
    generative_evaluator: Generative (diffusion) model evaluation pipeline.
    position_importance: Position-wise importance and attention analysis.
"""

from __future__ import annotations

# Lazy imports to avoid loading heavy dependencies at package import time.
__all__ = [
    # expression_comparison
    "compute_dynamic_range",
    "compute_fitness_gaps",
    "compute_uniformity_ks",
    "compute_ecdf_area_deviation",
    "full_statistical_report",
    # nrp_analysis
    "compute_non_repetitive_count",
    "monte_carlo_subsampling",
    "compute_kmer_entropy",
    "compute_pairwise_edit_distances",
    # dataset_evaluation
    "compute_dataset_statistics",
    "calculate_gc",
    "calculate_shannon_entropy",
    "calculate_position_information",
    # generative_evaluator
    "evaluate_generative_model",
    "compute_sequence_embeddings",
    "interpolate_sequences",
    "sample_sequences",
    "load_direct_diffusion_model",
    "DNADataset",
    # position_importance
    "evaluate_position_importance",
    "load_model_config",
    "load_model_from_checkpoint",
    "get_attention_weights_from_model",
    "compute_gradient_based_importance",
    "compute_position_importance",
    "save_position_importance_report",
    # predictor_inference
    "detect_biopart_from_path",
    "find_fold_checkpoints",
    "load_label_transform",
    "infer_model_hyperparams",
    "detect_task_type",
    "load_predictor",
    "ensemble_predict",
    "predict_with_sliding_window",
    "load_activity_transform",
    "apply_activity_transform",
    "inverse_log_label",
    "ornet_logits_to_probs",
    "get_default_model_dir",
    "compute_regression_metrics",
    "compute_classification_metrics",
]


def __getattr__(name: str):
    """Lazy-load evaluation functions on first access."""
    _expr = {
        "compute_dynamic_range",
        "compute_fitness_gaps",
        "compute_uniformity_ks",
        "compute_ecdf_area_deviation",
        "full_statistical_report",
    }
    _nrp = {
        "compute_non_repetitive_count",
        "monte_carlo_subsampling",
        "compute_kmer_entropy",
        "compute_pairwise_edit_distances",
    }
    _dataset = {
        "compute_dataset_statistics",
        "calculate_gc",
        "calculate_shannon_entropy",
        "calculate_position_information",
    }

    if name in _expr:
        from evaluation.expression_comparison import (
            compute_dynamic_range,
            compute_fitness_gaps,
            compute_uniformity_ks,
            compute_ecdf_area_deviation,
            full_statistical_report,
        )
        return locals()[name]

    if name in _nrp:
        from evaluation.nrp_analysis import (
            compute_non_repetitive_count,
            monte_carlo_subsampling,
            compute_kmer_entropy,
            compute_pairwise_edit_distances,
        )
        return locals()[name]

    if name in _dataset:
        from evaluation.dataset_evaluation import (
            compute_dataset_statistics,
            calculate_gc,
            calculate_shannon_entropy,
            calculate_position_information,
        )
        return locals()[name]

    _gen = {
        "evaluate_generative_model",
        "compute_sequence_embeddings",
        "interpolate_sequences",
        "sample_sequences",
        "load_direct_diffusion_model",
        "DNADataset",
    }

    if name in _gen:
        from evaluation.generative_evaluator import (
            evaluate_generative_model,
            compute_sequence_embeddings,
            interpolate_sequences,
            sample_sequences,
            load_direct_diffusion_model,
            DNADataset,
        )
        return locals()[name]

    _pos = {
        "evaluate_position_importance",
        "load_model_config",
        "load_model_from_checkpoint",
        "get_attention_weights_from_model",
        "compute_gradient_based_importance",
        "compute_position_importance",
        "save_position_importance_report",
    }

    if name in _pos:
        from evaluation.position_importance import (
            evaluate_position_importance,
            load_model_config,
            load_model_from_checkpoint,
            get_attention_weights_from_model,
            compute_gradient_based_importance,
            compute_position_importance,
            save_position_importance_report,
        )
        return locals()[name]

    _infer = {
        "detect_biopart_from_path", "find_fold_checkpoints", "load_label_transform",
        "infer_model_hyperparams", "detect_task_type", "load_predictor",
        "ensemble_predict", "predict_with_sliding_window",
        "load_activity_transform", "apply_activity_transform",
        "inverse_log_label", "ornet_logits_to_probs",
        "get_default_model_dir",
        "compute_regression_metrics", "compute_classification_metrics",
    }

    if name in _infer:
        from evaluation.predictor_inference import (
            detect_biopart_from_path, find_fold_checkpoints, load_label_transform,
            infer_model_hyperparams, detect_task_type, load_predictor,
            ensemble_predict, predict_with_sliding_window,
            load_activity_transform, apply_activity_transform,
            inverse_log_label, ornet_logits_to_probs,
            get_default_model_dir,
            compute_regression_metrics, compute_classification_metrics,
        )
        return locals()[name]

    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
