"""predict_evo_batch.py — Load the Evo model in evo-env and batch-predict sequence activity.

Invoked as a subprocess by ``scripts/design_iterative_library.py``'s
``predict_fitness_evo`` (under evo-env); reads sequences from a temp file and
writes predictions back.

Usage:
    conda activate evo-env
    cd <repo-root>
    python scripts/utils/predict_evo_batch.py \
        --input /tmp/seqs.txt --output /tmp/preds.csv \
        --checkpoint predictor_checkpoints/language_model/promoter_LoRA_finetune/best_model.pth \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# lora_finetune_evo requires "from src.utils..." to resolve
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Standalone Evo loading logic (visualization dependencies removed)
# ---------------------------------------------------------------------------

def _load_evo_base(local_model_path, device="cuda:0"):
    """Load the StripedHyena base model (HuggingFace sharded weights)."""
    import yaml
    from stripedhyena.utils import dotdict
    from stripedhyena.model import StripedHyena
    from stripedhyena.tokenizer import CharLevelTokenizer

    config_filename = "evo-1-8k-base_inference.yml"
    config_paths = [
        os.path.join(_PROJECT_ROOT, "src", "evo", "evo", "configs", config_filename),
        os.path.join(_PROJECT_ROOT, "src", "evo", "configs", config_filename),
    ]
    config_path = None
    for p in config_paths:
        if os.path.exists(p):
            config_path = p
            break
    if config_path is None:
        raise FileNotFoundError(f"Config {config_filename} not found")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    global_config = dotdict(config, Loader=yaml.FullLoader)
    sh_model = StripedHyena(global_config)

    # Load sharded weights
    index_file = os.path.join(local_model_path, "pytorch_model.bin.index.json")
    safetensors_file = os.path.join(local_model_path, "model.safetensors")
    pytorch_file = os.path.join(local_model_path, "pytorch_model.bin")

    if os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_file)
    elif os.path.exists(index_file):
        print(f"Loading sharded model from {index_file}")
        with open(index_file) as f:
            index = json.load(f)
        shards_cache = {}
        state_dict = {}
        for key, filename in index["weight_map"].items():
            if filename not in shards_cache:
                shards_cache[filename] = torch.load(
                    os.path.join(local_model_path, filename), map_location="cpu"
                )
            state_dict[key] = shards_cache[filename][key]
    elif os.path.exists(pytorch_file):
        state_dict = torch.load(pytorch_file, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights in {local_model_path}")

    # Strip the "backbone." prefix
    if all(k.startswith("backbone.") for k in state_dict):
        state_dict = {k.replace("backbone.", "", 1): v for k, v in state_dict.items()}

    sh_model.load_state_dict(state_dict, strict=True)
    sh_model.to_bfloat16_except_poles_residues()
    sh_model = sh_model.to(device)

    tokenizer = CharLevelTokenizer(512)

    class _LocalEvo:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

    return _LocalEvo(sh_model, tokenizer), tokenizer


def _load_finetuned(checkpoint_path, evo_model_path, device="cuda:0"):
    """Load the fine-tuned Evo model (base + LoRA + regression head)."""
    # Import lora_finetune_evo
    finetune_path = os.path.join(_PROJECT_ROOT, "src", "evo", "lora_finetune_evo.py")
    spec = importlib.util.spec_from_file_location("lora_finetune_evo", finetune_path)
    finetune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finetune)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    label_transform = ckpt.get("label_transform", None)

    lora_r, hidden_dim = None, None
    for key, val in state_dict.items():
        if "lora_A" in key and lora_r is None:
            lora_r = val.shape[0]
        if "head.regression.0.weight" in key and hidden_dim is None:
            hidden_dim = val.shape[0]
    lora_r = lora_r or 32
    hidden_dim = hidden_dim or 512
    print(f"  lora_r={lora_r}, hidden_dim={hidden_dim}, label_transform={label_transform}")

    evo_obj, tokenizer = _load_evo_base(evo_model_path, device)

    model = finetune.EvoWithRegressionHead(
        evo_obj,
        hidden_dim=hidden_dim,
        dropout=0.2,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.1,
    ).to(device)

    model_sd = model.state_dict()
    new_sd = {}
    for key, val in state_dict.items():
        new_key = key.replace("regression_head.", "head.", 1) if key.startswith("regression_head.") else key
        if new_key in model_sd:
            new_sd[new_key] = val
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    print(f"  epoch={ckpt.get('epoch', '?')}, R²={ckpt.get('val_r2', '?')}")
    return model, tokenizer, label_transform


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, sequences, tokenizer, device, batch_size=16):
    """Batch prediction."""
    finetune_path = os.path.join(_PROJECT_ROOT, "src", "evo", "lora_finetune_evo.py")
    spec = importlib.util.spec_from_file_location("_ft", finetune_path)
    _ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_ft)

    from torch.utils.data import DataLoader
    dataset = _ft.PromoterDataset(sequences, [0] * len(sequences), tokenizer, task_type="regression")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                input_ids, _, attn_mask = batch
            else:
                input_ids, _ = batch
                attn_mask = None
            input_ids = input_ids.to(device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            p = model(input_ids, attn_mask)
            preds.extend(p.cpu().numpy())

    return np.array(preds).flatten()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evo_model", default=os.path.join(
        _PROJECT_ROOT, "src", "evo", "models", "evo-1.5-8k-base"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print(f"[predict_evo_batch] {args.input}")
    with open(args.input) as f:
        sequences = [line.strip().upper() for line in f if line.strip()]
    print(f"[predict_evo_batch] {len(sequences)} sequences")

    t0 = time.time()
    model, tokenizer, label_transform = _load_finetuned(
        args.checkpoint, args.evo_model, args.device,
    )
    print(f"[predict_evo_batch] Loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    predictions = predict(model, sequences, tokenizer, args.device, args.batch_size)
    print(f"[predict_evo_batch] Predicted in {time.time() - t0:.1f}s")

    if label_transform == "log10":
        predictions = np.power(10, predictions) - 1.0

    with open(args.output, "w") as f:
        f.write("index,predicted_activity\n")
        for i, val in enumerate(predictions):
            f.write(f"{i},{val:.6f}\n")
    print(f"[predict_evo_batch] Saved to {args.output}")


if __name__ == "__main__":
    main()
