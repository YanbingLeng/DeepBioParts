"""Evo 1.5 ablation workflow (library layer).

Orchestration for the Evo ablation pipeline: fixed external test set + manifest
five-fold split + training + ensemble inference on the fixed test set + canonical
artifact writing. All heavyweight Evo dependencies (torch / Evo weights) are
imported lazily inside this module, keeping the evo-env dependency local and
avoiding any import-time cost.

Design principle: functions accept explicit parameters only and never depend on
an argparse namespace, keeping them decoupled from the CLI.
"""

import gc
import os

import numpy as np

from src.data.ablation_split import (
    exclude_test_from_train,
    get_or_create_manifest,
    load_fixed_test_set,
)
from src.evaluation.ablation_artifacts import (
    write_ablation_test_results,
    write_fold_metrics,
    write_parameter_counts,
)

__all__ = ["run_evo_ablation", "evo_ablation_test_inference"]


def run_evo_ablation(
    *,
    biopart: str,
    task_type: str,
    data_path: str,
    output_dir: str,
    max_len: int,
    evo_adaptation: str,
    pooling_mode: str,
    n_folds: int,
    single_split: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    device: str,
    gradient_checkpointing: bool,
    gradient_accumulation_steps: int,
    similarity_threshold: float,
    kmer_size: int,
    use_log_label: bool,
    fixed_test_dir: str,
    ablation_manifest_dir: str,
):
    """Evo ablation workflow: fixed test set + manifest fold split + ensemble inference on the fixed test set.

    Called when running Evo ablations in an actual GPU environment.
    """
    import pandas as pd
    # Evo training + inference run in the same process: enable expandable_segments to reduce
    # fragmentation and avoid OOM when the backbone is reloaded for inference
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from src.evo.lora_finetune_evo import finetune_evo_promoter_strength

    # 1) Fixed test set
    test_seqs, test_labels, meta_df, _ = load_fixed_test_set(biopart, task_type, fixed_test_dir)

    # 2) Exclude overlap from the training pool and persist the cleaned CSV for finetune to read
    df = pd.read_csv(data_path)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "sequence":
            rename[c] = "sequence"
        elif cl in ("activity", "label"):
            rename[c] = "activity"
    df = df.rename(columns=rename)
    pool_seqs, pool_labels = exclude_test_from_train(
        df["sequence"].astype(str).tolist(), df["activity"].values, test_seqs
    )
    cleaned_csv = os.path.join(output_dir, "cleaned_train.csv")
    pd.DataFrame({"sequence": pool_seqs, "activity": pool_labels}).to_csv(cleaned_csv, index=False)

    # 3) Manifest fold split
    fold_assignment = get_or_create_manifest(
        biopart, task_type, pool_seqs, pool_labels, test_seqs,
        similarity_threshold=similarity_threshold,
        kmer_size=kmer_size,
        n_folds=n_folds,
        manifest_dir=ablation_manifest_dir,
    )
    print(f"\n[Ablation-Evo] biopart={biopart} task={task_type} "
          f"adapt={evo_adaptation} pool={pooling_mode} | "
          f"test={len(test_seqs)} train_pool={len(pool_seqs)}")

    # 4) Training (forced K-Fold + fixed test set + precomputed fold assignment)
    result = finetune_evo_promoter_strength(
        data_path=cleaned_csv,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        dropout=lora_dropout,
        use_lora=(evo_adaptation == "lora"),
        lora_r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
        max_len=max_len,
        early_stopping_patience=patience,
        task_type=task_type,
        use_gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_similarity_split=False,  # fold split is provided by the manifest
        similarity_threshold=similarity_threshold,
        kmer_size=kmer_size,
        n_folds=n_folds,
        use_kfold_cv=True,
        use_log_label=use_log_label,
        evo_adaptation=evo_adaptation,
        pooling_mode=pooling_mode,
        fixed_test_seqs=test_seqs,
        fixed_test_labels=test_labels,
        fold_assignment=fold_assignment,
        single_split=single_split,
    )
    fold_model_paths = result[0] if result else []
    fold_val_scores = result[1] if result else []

    # 5) Ensemble inference on the fixed test set + write canonical outputs
    per_fold, ensemble = evo_ablation_test_inference(
        fold_model_paths, test_seqs, max_len, task_type,
        evo_adaptation, pooling_mode, device, use_log_label, output_dir,
    )
    # Classification ablations keep the fold_metrics column name val_f1; regression uses val_pearson_r
    _fold_metric = "val_pearson_r" if task_type == "regression" else "val_f1"
    write_fold_metrics(output_dir, fold_val_scores, task_type, metric_name=_fold_metric)
    write_ablation_test_results(
        output_dir, meta_df, test_seqs, np.asarray(test_labels),
        per_fold, np.asarray(ensemble), task_type,
    )
    return result


def evo_ablation_test_inference(fold_model_paths, test_seqs, max_len, task_type,
                                evo_adaptation, pooling_mode, device, use_log_label,
                                output_dir):
    """Load each fold's Evo model, run inference on the fixed test set, and ensemble
    (regression: mean over folds; classification: argmax).

    Retains GPU memory hygiene: the GPU is fully released after training, the same
    backbone weights are reused to avoid reloading the 6.5B parameters, and each fold's
    model is followed by ``del model`` + ``empty_cache``.
    """
    import torch
    from src.evo.lora_finetune_evo import (
        EvoWithRegressionHead,
        dna_sequence_to_tokens,
        load_evo_from_local,
    )

    # Training has just finished: fully release the GPU first (avoid OOM when reloading the 6.5B backbone due to fragmentation)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PROJECT_ROOT = src/evo/../.. ; locate the local Evo weights directory relative to this file
    evo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src/evo
    local_model_path = str(os.path.join(evo_dir, "models", "evo-1.5-8k-base"))
    # Load directly onto the target device to avoid a second CPU->GPU transfer and fragmentation
    shared = load_evo_from_local(local_model_path, "evo-1.5-8k-base", device)
    tokenizer = shared.tokenizer
    pad_id = getattr(tokenizer, "pad_id", getattr(tokenizer, "pad_token_id", 0))

    # Tokenize + attention mask (consistent with PromoterDataset)
    tokens_list, mask_list = [], []
    for seq in test_seqs:
        toks = tokenizer.tokenize(seq)[:max_len]
        real_len = len(toks)
        toks = toks + [pad_id] * (max_len - len(toks))
        mask = [1] * real_len + [0] * (max_len - real_len)
        tokens_list.append(toks)
        mask_list.append(mask)
    tokens_t = torch.tensor(tokens_list, dtype=torch.long)
    mask_t = torch.tensor(mask_list, dtype=torch.long)

    all_preds = []
    counts_written = False
    batch = 8  # conservative batch size to limit GPU memory pressure during inference
    for p in fold_model_paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        ea = ck.get("evo_adaptation", evo_adaptation)
        pm = ck.get("pooling_mode", pooling_mode)
        # Infer hidden_dim / lora_r from the state_dict
        hidden_dim = 512
        for k in sd:
            if "head.regression.0.weight" in k or "head.classification.0.weight" in k:
                hidden_dim = sd[k].shape[0]
                break
        lora_keys = [k for k in sd if "lora_A" in k and "blocks.0" in k]
        lora_r = sd[lora_keys[0]].shape[0] if lora_keys else 16

        model = EvoWithRegressionHead(
            shared, hidden_dim=hidden_dim, dropout=0.2,
            use_lora=(ea == "lora"), lora_r=lora_r, lora_alpha=lora_r * 2 if lora_r else 32,
            lora_dropout=0.1, task_type=task_type,
            evo_adaptation=ea, pooling_mode=pm,
        ).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()

        if not counts_written:
            write_parameter_counts(
                {"total": sum(p2.numel() for p2 in model.parameters()),
                 "trainable": sum(p2.numel() for p2 in model.parameters() if p2.requires_grad),
                 "evo_adaptation": ea, "pooling_mode": pm},
                output_dir,
            )
            counts_written = True

        fold_preds = []
        with torch.no_grad():
            for i in range(0, len(tokens_t), batch):
                ti = tokens_t[i:i + batch].to(device)
                mi = mask_t[i:i + batch].to(device)
                out = model(ti, mi)
                if task_type == "classification":
                    fold_preds.append(torch.argmax(out, dim=1).cpu().numpy())
                else:
                    fold_preds.append(out.cpu().numpy())
        all_preds.append(np.concatenate([np.atleast_1d(x) for x in fold_preds]))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Release the shared backbone
    del shared
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_fold = all_preds
    stacked = np.array(all_preds)
    if task_type == "regression":
        ensemble = np.mean(stacked, axis=0)
        if use_log_label:
            ensemble = np.power(10.0, ensemble) - 1.0
            per_fold = [np.power(10.0, f) - 1.0 for f in per_fold]
    else:
        ensemble = np.round(np.mean(stacked, axis=0)).astype(int)
    return per_fold, ensemble
