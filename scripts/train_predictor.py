#!/usr/bin/env python3
"""Unified predictor training for DeepBioParts.

A single entry point that trains any of the 4 predictor model types:
  - attnbilstm   (Attention + Bidirectional LSTM)
  - 1dcnn        (1D-CNN, suited for short sequences)
  - evo          (Evo 1.5 DNA foundation model + LoRA fine-tuning)

This script is a thin CLI entry point: argument parsing + dispatch. Training logic
lives in ``src/training/predictor.py`` (``Predictor_language``), ablation artifact
writing in ``src/evaluation/ablation_artifacts.py``, the Evo ablation workflow in
``src/evo/ablation.py``, training-curve plotting in ``src/visualization``, and
training defaults in ``src/config/constants.py``.

Usage examples:
    # CNN–Attention–BiLSTM with 5-fold CV (cluster-based split by default)
    python scripts/train_predictor.py --model_type attnbilstm --biopart promoter

    # 1D-CNN for RBS with 3-fold CV
    python scripts/train_predictor.py --model_type 1dcnn --biopart rbs --n_folds 3

    # Terminator with log10 label transform
    python scripts/train_predictor.py --model_type attnbilstm --biopart terminator --log_label

    # Evo with LoRA fine-tuning
    python scripts/train_predictor.py --model_type evo --biopart promoter --n_folds 5 --lora_rank 16
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Resolve the project root directory and add it to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config.constants import (
    NAIVE_DEFAULT_PARAMS,
    RBS_DEFAULT_PARAMS,
    get_biopart_defaults,
)
from src.evaluation.ablation_artifacts import (
    ablation_run_dir,
    is_ablation,
    variant_tag,
    write_ablation_test_results,
    write_fold_metrics,
    write_parameter_counts,
    write_run_config,
)
from src.visualization import (
    EVO_CLASSIFICATION_LAYOUT,
    EVO_REGRESSION_LAYOUT,
    NAIVE_CLASSIFICATION_LAYOUT,
    NAIVE_REGRESSION_LAYOUT,
    plot_per_fold_training_curves,
)



# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified predictor training for DeepBioParts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required ----
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["attnbilstm", "1dcnn", "evo"],
                        help="Model architecture")
    parser.add_argument("--biopart", type=str, required=True,
                        choices=["promoter", "rbs", "terminator"],
                        help="Biological part type")

    # ---- Common training parameters ----
    parser.add_argument("--data_path", type=str, default=None,
                        help="Training data CSV (default: auto-detect by biopart)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./predictor_checkpoints/)")
    parser.add_argument("--task_type", type=str, default="regression",
                        choices=["regression", "classification"])
    parser.add_argument("--epochs", type=int, default=500,
                        help="Max training epochs (default: 500)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: 64 naive / 32 evo)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 1e-4 naive / 5e-5 evo)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (default: 100 naive / 15 evo)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag appended to output model name (default: auto-generated timestamp)")
    parser.add_argument("--device", type=str, default="cuda:0")

    # ---- Cross-validation ----
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of k-fold CV folds")
    parser.add_argument("--no_kfold", action="store_true",
                        help="Disable k-fold, use single train/val split")
    parser.add_argument("--log_label", action="store_true",
                        help="Apply log10(label+1) transform to labels (for terminator regression)")

    # ---- Visualization & Logging ----

    # ---- Naive model specific ----
    naive_group = parser.add_argument_group("Naive model options")
    naive_group.add_argument("--encoding_type", type=str, default="onehot",
                             choices=["onehot", "single", "pairs", "triples"])
    naive_group.add_argument("--use_cluster_split", action="store_true",
                             help="Use similarity-based cluster splitting")

    # ---- Data augmentation ----
    aug_group = parser.add_argument_group("Data augmentation")
    aug_group.add_argument("--mixup", action="store_true",
                           help="Enable Mixup data augmentation (batch-level regularization during training)")
    aug_group.add_argument("--mixup_alpha", type=float, default=0.2,
                           help="Mixup Beta distribution parameter (default: 0.2)")
    aug_group.add_argument("--mutation_augment", action="store_true",
                           help="Enable random mutation augmentation (generates virtual neighbor samples)")
    aug_group.add_argument("--mutation_n", type=int, default=1,
                           help="Number of mutation positions per sequence (default: 1)")
    aug_group.add_argument("--mutation_copies", type=int, default=1,
                           help="Mutation augmentation copy count, 1 = double, 2 = triple (default: 1)")
    aug_group.add_argument("--neighbor_interp", action="store_true",
                           help="Enable Hamming=1 neighbor linear interpolation augmentation")
    aug_group.add_argument("--interp_lambdas", type=float, nargs="+", default=[0.5],
                           help="Interpolation lambda list (default: 0.5)")
    aug_group.add_argument("--interp_max_hamming", type=int, default=1,
                           help="Maximum Hamming distance for interpolation neighbors (default: 1)")

    # ---- Model architecture overrides ----
    arch_group = parser.add_argument_group("Model architecture overrides")
    arch_group.add_argument("--dropout_rate", type=float, default=None,
                            help="Dropout rate (default: RBS=0.5, others=0.2)")
    arch_group.add_argument("--conv_hidden", type=int, default=None,
                            help="Conv hidden channels (default: RBS=32, others=128)")
    arch_group.add_argument("--motif_conv_hidden", type=int, default=None,
                            help="Motif conv hidden channels (default: RBS=64, others=256)")
    arch_group.add_argument("--n_heads", type=int, default=None,
                            help="Attention heads (default: RBS=2, others=16)")
    arch_group.add_argument("--conv_width_motif", type=int, default=None,
                            help="Conv kernel width (default: RBS=3, others=5)")
    arch_group.add_argument("--weight_decay", type=float, default=None,
                            help="Weight decay (default: RBS=1e-3, others=1e-4)")

    # ---- Evo model specific ----
    evo_group = parser.add_argument_group("Evo model options")
    evo_group.add_argument("--lora_rank", type=int, default=16)
    evo_group.add_argument("--lora_alpha", type=int, default=32)
    evo_group.add_argument("--no_lora", action="store_true")
    evo_group.add_argument("--gradient_checkpointing", action="store_true")
    evo_group.add_argument("--hidden_dim", type=int, default=512,
                           help="Regression head hidden dim")
    evo_group.add_argument("--similarity_threshold", type=float, default=0.8)
    evo_group.add_argument("--kmer_size", type=int, default=3)
    evo_group.add_argument("--gradient_accumulation_steps", type=int, default=2)
    evo_group.add_argument("--lora_dropout", type=float, default=0.1)

    # ---- Ablation experiment ----
    abl_group = parser.add_argument_group("Ablation experiment")
    abl_group.add_argument("--ablation_variant", type=str, default=None,
                           choices=["full", "no_attention", "no_bilstm", "no_attention_no_bilstm"],
                           help="CNN–Attention–BiLSTM ablation variant (attnbilstm/1dcnn only). Passing any ablation flag enables fixed-external-test-set mode")
    abl_group.add_argument("--evo_adaptation", type=str, default=None,
                           choices=["lora", "head_only", "partial_ft"],
                           help="Evo backbone adaptation (evo only). full=lora+attention, head_only, mean_pool=lora+mean arise from combinations")
    abl_group.add_argument("--pooling_mode", type=str, default=None,
                           choices=["attention", "mean"],
                           help="Evo pooling mode (evo only)")
    abl_group.add_argument("--fixed_test_dir", type=str, default="results/test_set_results",
                           help="Fixed external test set directory (read-only; never recomputed or overwritten)")
    abl_group.add_argument("--ablation_manifest_dir", type=str,
                           default="results/test_set_results/ablation_manifests",
                           help="Directory of cluster-based five-fold split manifests (reused across variants to guarantee an identical split)")
    abl_group.add_argument("--ablation_output_root", type=str,
                           default="predictor_checkpoints/ablation",
                           help="Root output directory for ablation experiments")
    abl_group.add_argument("--single_split", action="store_true",
                           help="In ablation mode, skip K-fold CV: train a single model with fold==1 as validation (5x speed-up; statistical units become seeds)")

    args = parser.parse_args()
    _validate_ablation_args(args, parser)
    return args


def _validate_ablation_args(args, parser):
    """Validate legal combinations of ablation flags for the given model_type."""
    if args.model_type == "evo":
        if args.ablation_variant is not None:
            parser.error("--ablation_variant applies only to --model_type attnbilstm/1dcnn and cannot be combined with evo")
    else:
        if args.evo_adaptation is not None or args.pooling_mode is not None:
            parser.error("--evo_adaptation / --pooling_mode apply only to --model_type evo")


# ---------------------------------------------------------------------------
# Helpers (CLI boundary: translate args into explicit library-level parameters)
# ---------------------------------------------------------------------------

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ablation_config(args, extra: dict = None) -> dict:
    """Build the run_config.json configuration dict from args (decoupling boundary between CLI and library)."""
    cfg = {
        "model_type": args.model_type,
        "biopart": args.biopart,
        "task_type": args.task_type,
        "ablation_variant": args.ablation_variant,
        "evo_adaptation": args.evo_adaptation,
        "pooling_mode": args.pooling_mode,
        "variant_tag": variant_tag(args.model_type, args.ablation_variant,
                                   args.evo_adaptation, args.pooling_mode),
        "n_folds": 1 if args.no_kfold else args.n_folds,
        "fixed_test_dir": args.fixed_test_dir,
        "ablation_manifest_dir": args.ablation_manifest_dir,
        "data_path": args.data_path,
        "seed": args.seed,
    }
    if extra:
        cfg.update(extra)
    return cfg


def _ablation_output_dir(args, timestamp) -> str:
    """Default seed-stable output directory in ablation mode; --output_dir takes precedence."""
    vtag = variant_tag(args.model_type, args.ablation_variant,
                       args.evo_adaptation, args.pooling_mode)
    return ablation_run_dir(args.biopart, args.task_type, vtag, timestamp,
                            args.ablation_output_root)


def _plot_naive_curves(predictor, savepath, task_type):
    """Convert Predictor_language.fold_metrics into per-fold histories and call the unified plotting function."""
    fold_metrics = getattr(predictor, "fold_metrics", None)
    if not fold_metrics:
        print("Warning: No fold metrics found; skipping training-curve plot")
        return
    histories = []
    for fold_idx in sorted(fold_metrics.keys()):
        epoch_dicts = fold_metrics[fold_idx]
        keys = set().union(*epoch_dicts) if epoch_dicts else set()
        histories.append({k: [d.get(k, float("nan")) for d in epoch_dicts] for k in keys})
    layout = NAIVE_REGRESSION_LAYOUT if task_type == "regression" else NAIVE_CLASSIFICATION_LAYOUT
    plot_per_fold_training_curves(histories, savepath, layout)


# ---------------------------------------------------------------------------
# Naive model dispatch
# ---------------------------------------------------------------------------

def train_naive_model(args):
    """Train attnbilstm or 1dcnn via Predictor_language."""
    import pandas as pd
    from src.training.predictor import Predictor_language

    defaults = get_biopart_defaults(args.biopart, args.task_type)
    data_path = args.data_path or defaults["data_path"]
    if not data_path:
        raise SystemExit("--data_path is required (no default dataset path is bundled)")
    seq_len = defaults["seq_len"]
    patience = args.patience or defaults["patience"]

    # Select hyperparameter set based on biopart
    params = dict(RBS_DEFAULT_PARAMS if args.biopart == "rbs" else NAIVE_DEFAULT_PARAMS)

    # CLI overrides
    if args.lr is not None:
        params["learning_rate"] = args.lr
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    for k in ("dropout_rate", "conv_hidden", "motif_conv_hidden", "n_heads", "conv_width_motif", "weight_decay"):
        v = getattr(args, k, None)
        if v is not None:
            params[k] = v
    epoch = args.epochs

    # Augmentation defaults from BIOPART_DEFAULTS (CLI flags override)
    use_neighbor_interp = args.neighbor_interp or defaults.get("neighbor_interp", False)
    interp_lambdas = args.interp_lambdas if args.interp_lambdas != [0.5] else defaults.get("interp_lambdas", [0.5])
    use_mutation_augment = args.mutation_augment or defaults.get("mutation_augment", False)
    mutation_n = args.mutation_n if args.mutation_n != 1 else defaults.get("mutation_n", 1)
    mutation_copies = args.mutation_copies if args.mutation_copies != 1 else defaults.get("mutation_copies", 1)
    use_log_label = args.log_label or defaults.get("log_label", False)
    use_cluster_split = args.use_cluster_split or defaults.get("use_cluster_split", False)

    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    seqs = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].astype(float).values.tolist()

    # Output path with date subdirectory
    timestamp = datetime.now().strftime('%Y%m%d')
    tag_suffix = f"_{args.tag}" if args.tag else ""
    model_name = f"{args.biopart}_{args.model_type}_{args.encoding_type}{tag_suffix}_{timestamp}"

    abl = is_ablation(args.ablation_variant, args.evo_adaptation, args.pooling_mode)
    if abl:
        savepath = args.output_dir or _ablation_output_dir(args, timestamp)
    elif args.output_dir:
        savepath = args.output_dir
    else:
        savepath = str(PROJECT_ROOT / "predictor_checkpoints" / "supervised_model" / model_name)
    os.makedirs(savepath, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model:       {args.model_type}")
    print(f"Biopart:     {args.biopart}")
    print(f"Encoding:    {args.encoding_type}")
    print(f"Seq length:  {seq_len}")
    print(f"N samples:   {len(seqs)}")
    print(f"N folds:     {'single split' if args.no_kfold else args.n_folds}")
    print(f"Log label:   {use_log_label}")
    print(f"Max epochs:  {epoch}")
    print(f"{'='*60}\n")

    # Create predictor
    predictor = Predictor_language(
        seq_len=seq_len,
        model_type=args.model_type,
        model_name=model_name,
        encoding_type=args.encoding_type,
        task_type=args.task_type,
        batch_size=params["batch_size"],
        epoch=epoch,
        patience=patience,
        learning_rate=params["learning_rate"],
        dropout_rate=params["dropout_rate"],
        conv_width_motif=params["conv_width_motif"],
        n_heads=params["n_heads"],
        conv_hidden=params["conv_hidden"],
        motif_conv_hidden=params["motif_conv_hidden"],
        weight_decay=params["weight_decay"],
        optimizer=params["optimizer"],
        use_cluster_split=use_cluster_split,
        n_folds=1 if args.no_kfold else args.n_folds,
        use_log_label=use_log_label,
        use_mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        use_mutation_augment=use_mutation_augment,
        mutation_n=mutation_n,
        mutation_copies=mutation_copies,
        use_neighbor_interp=use_neighbor_interp,
        interp_lambdas=interp_lambdas,
        interp_max_hamming=args.interp_max_hamming,
        ablation_variant=args.ablation_variant or "full",
        seed=args.seed,
    )

    # ===== Ablation mode: fixed external test set + precomputed fold assignment =====
    if abl:
        from src.data.ablation_split import (
            exclude_test_from_train,
            get_or_create_manifest,
            load_fixed_test_set,
        )
        # The ablation protocol is fixed to cluster-based K-Fold; --no_kfold has no effect in ablation mode
        if args.no_kfold:
            print("[Ablation] The ablation protocol enforces cluster-based K-Fold; ignoring --no_kfold")
        test_seqs, test_labels, meta_df, _ = load_fixed_test_set(
            args.biopart, args.task_type, args.fixed_test_dir
        )
        # Exclude sequences overlapping the fixed test set from the training pool (validated internally)
        pool_seqs, pool_labels = exclude_test_from_train(seqs, labels, test_seqs)
        n_folds_eff = max(2, args.n_folds)
        predictor.n_folds = n_folds_eff  # align with the manifest fold count (overrides any --no_kfold)
        fold_assignment = get_or_create_manifest(
            args.biopart, args.task_type, pool_seqs, pool_labels, test_seqs,
            similarity_threshold=args.similarity_threshold,
            kmer_size=args.kmer_size,
            n_folds=n_folds_eff,
            manifest_dir=args.ablation_manifest_dir,
        )
        print(f"\n[Ablation] biopart={args.biopart} task={args.task_type} "
              f"variant={variant_tag(args.model_type, args.ablation_variant, args.evo_adaptation, args.pooling_mode)} "
              f"| test={len(test_seqs)} train_pool={len(pool_seqs)}")

        # Train (passing in the fixed test set and fold assignment)
        predictor.train(
            pool_seqs, pool_labels, savepath,
            fixed_test=(test_seqs, test_labels),
            fold_assignment=fold_assignment,
            single_split=args.single_split,
        )

        # Run inference on the fixed test set and write canonical outputs
        test_dir = os.path.join(savepath, "test_results")
        # For classification, test() returns class-1 probabilities (per-fold + ensemble)
        # for downstream AUROC computation
        per_fold, ensemble, _ = predictor.test(save_dir=test_dir, return_predictions=True)
        write_fold_metrics(savepath, getattr(predictor, "fold_val_scores", []), args.task_type)
        write_ablation_test_results(
            savepath, meta_df, test_seqs, np.asarray(test_labels),
            per_fold, np.asarray(ensemble), args.task_type,
        )
        write_run_config(savepath, _ablation_config(args, extra={"model_name": model_name}))
        # Parameter counts
        # Use the facade (src/models/__init__.py) to avoid double-loading
        # src.models.predictor and models.predictor, which would make
        # @ModelFactory.register print twice.
        from src.models import ModelFactory
        _m = ModelFactory.create(
            args.model_type, seq_len=seq_len, num_classes=predictor.num_classes,
            dropout_rate=predictor.dropout_rate, conv_width_motif=predictor.conv_width_motif,
            n_heads=predictor.n_heads, conv_hidden=predictor.conv_hidden,
            motif_conv_hidden=predictor.motif_conv_hidden, vocab_size=predictor.vocab_size,
            task_type=args.task_type, ablation_variant=args.ablation_variant or "full",
        )
        write_parameter_counts(
            {"total": sum(p.numel() for p in _m.parameters()),
             "trainable": sum(p.numel() for p in _m.parameters() if p.requires_grad)},
            savepath,
        )
        _plot_naive_curves(predictor, savepath, args.task_type)
        print(f"\nAblation run complete. Outputs saved to: {savepath}")
        return

    # Train (get test data for later evaluation)
    predictor.train(seqs, labels, savepath)

    # Test
    test_dir = os.path.join(savepath, 'test_results')
    predictor.test(save_dir=test_dir)

    # Plot training curves (Nature-style SVG)
    _plot_naive_curves(predictor, savepath, args.task_type)

    print(f"\nTraining complete. Model saved to: {savepath}")


# ---------------------------------------------------------------------------
# Evo model dispatch
# ---------------------------------------------------------------------------

def train_evo_model(args):
    """Train Evo with LoRA fine-tuning."""
    from src.evo.lora_finetune_evo import finetune_evo_promoter_strength

    defaults = get_biopart_defaults(args.biopart, args.task_type)
    data_path = args.data_path or defaults["data_path"]
    if not data_path:
        raise SystemExit("--data_path is required (no default dataset path is bundled)")
    max_len = defaults["seq_len"]

    # Output path with date subdirectory
    timestamp = datetime.now().strftime('%Y%m%d')
    tag_suffix = f"_{args.tag}" if args.tag else ""
    model_name = f"{args.biopart}_evo_lora{tag_suffix}_{timestamp}"

    abl = is_ablation(args.ablation_variant, args.evo_adaptation, args.pooling_mode)
    if abl:
        output_dir = args.output_dir or _ablation_output_dir(args, timestamp)
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(PROJECT_ROOT / "predictor_checkpoints" / "language_model" / model_name)
    os.makedirs(output_dir, exist_ok=True)

    use_lora = not args.no_lora
    use_kfold = not args.no_kfold
    use_log_label = args.log_label or defaults.get("log_label", False)
    use_cluster_split = args.use_cluster_split or defaults.get("use_cluster_split", False)
    # Evo leverages its pretrained genomic prior and uses no data augmentation
    # (augmentation inflates the sample count and would OOM the cluster similarity matrix);
    # augmentation arguments are therefore not forwarded, keeping the
    # finetune_evo_promoter_strength defaults (all off).

    print(f"\n{'='*60}")
    print(f"Model:       Evo (LoRA fine-tuning)")
    print(f"Biopart:     {args.biopart}")
    print(f"Max len:     {max_len}")
    print(f"N folds:     {'single split' if args.no_kfold else args.n_folds}")
    print(f"Log label:   {use_log_label}")
    print(f"LoRA:        {use_lora} (rank={args.lora_rank}, alpha={args.lora_alpha})")
    print(f"Cluster split: {use_cluster_split}")

    # ===== Ablation mode: fixed external test set + precomputed fold assignment =====
    if abl:
        from src.evo.ablation import run_evo_ablation
        result = run_evo_ablation(
            biopart=args.biopart, task_type=args.task_type, data_path=data_path,
            output_dir=output_dir, max_len=max_len,
            evo_adaptation=args.evo_adaptation or "lora",
            pooling_mode=args.pooling_mode or "attention",
            n_folds=args.n_folds, single_split=args.single_split,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, hidden_dim=args.hidden_dim,
            learning_rate=args.lr or 5e-5, batch_size=args.batch_size or 32,
            num_epochs=args.epochs, patience=args.patience or 15, device=args.device,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            similarity_threshold=args.similarity_threshold, kmer_size=args.kmer_size,
            use_log_label=use_log_label,
            fixed_test_dir=args.fixed_test_dir,
            ablation_manifest_dir=args.ablation_manifest_dir,
        )
        if result is not None:
            write_run_config(output_dir, _ablation_config(
                args, extra={"model_name": model_name, "max_len": max_len,
                             "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha}))
        print(f"\nAblation run complete. Outputs saved to: {output_dir}")
        return

    # Run training and get fold histories
    result = finetune_evo_promoter_strength(
        data_path=data_path,
        output_dir=output_dir,
        batch_size=args.batch_size or 32,
        num_epochs=args.epochs,
        learning_rate=args.lr or 5e-5,
        hidden_dim=args.hidden_dim,
        dropout=args.lora_dropout,
        use_lora=use_lora,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=args.device,
        max_len=max_len,
        early_stopping_patience=args.patience or 15,
        task_type=args.task_type,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_similarity_split=use_cluster_split,
        similarity_threshold=args.similarity_threshold,
        kmer_size=args.kmer_size,
        n_folds=args.n_folds,
        use_kfold_cv=use_kfold,
        use_log_label=use_log_label,
        evo_adaptation=args.evo_adaptation,
        pooling_mode=args.pooling_mode or "attention",
    )

    # Extract fold_histories from result (only available in K-Fold mode)
    fold_histories = None
    if use_kfold and result is not None and len(result) >= 6:
        fold_histories = result[5]

    # Plot training curves
    if fold_histories:
        layout = EVO_REGRESSION_LAYOUT if args.task_type == "regression" else EVO_CLASSIFICATION_LAYOUT
        plot_per_fold_training_curves(fold_histories, output_dir, layout)

    print(f"\nTraining complete. Checkpoints: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _set_seed(args.seed)

    if args.model_type == "evo":
        train_evo_model(args)
    else:
        train_naive_model(args)


if __name__ == "__main__":
    main()
