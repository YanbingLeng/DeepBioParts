#!/usr/bin/env python3
"""Train a direct one-hot DDPM (SimplexDenoiser) as the "naive DDPM" baseline in the Fig. 3b
generative-model ablation.

This model diffuses directly on the one-hot probability simplex, with **no VAE and no latent
space**, completing the PWM / VAE / LDM comparison: VAE (no diffusion) -> DDPM (direct diffusion,
no latent space) -> LDM (latent-space diffusion), thereby isolating the contribution of the
latent space to generation quality.

## Diffusion conventions (must match sampling in src/evaluation/generative_evaluator.py)
- VP linear noise schedule, eps-prediction (the model predicts the noise eps).
- Forward: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*eps
- Loss: MSE(noise_pred, eps)
- Decoding: argmax over the 4 channels in TCGA order (T=0, C=1, G=2, A=3), consistent with seq2onehot.

## Checkpoint layout (aligned with load_direct_diffusion_model)
    <output_dir>/config.json            # num_timesteps / beta_start / beta_end ...
    <output_dir>/checkpoints/best.pth   # {"model_state_dict": ..., "epoch": ..., "val_loss": ...}

Usage:
    python scripts/train_ddpm.py --biopart promoter --device cuda:0
    python scripts/train_ddpm.py --biopart rbs --epochs 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Path setup: src/ must be on sys.path because models.diffusion uses top-level `models.` imports internally.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.diffusion.denoiser import get_direct_denoiser_model  # noqa: E402
from models.diffusion.noise_schedule import VPNoiseSchedule  # noqa: E402

# TCGA channel order: T=0, C=1, G=2, A=3 (consistent with seq2onehot / sample_sequences decoding).
BASE_TO_CHAN = {"T": 0, "C": 1, "G": 2, "A": 3}

# Training data and sequence length per biopart.
BIOPART_DATA = {
    "promoter": {"seq_len": 40},
    "rbs": {"seq_len": 15},
    "terminator": {"seq_len": 50},
}


def encode_onehot(seqs: list[str], seq_len: int) -> np.ndarray:
    """Encode DNA sequences as [N, 4, L] one-hot arrays (TCGA channel order)."""
    arr = np.zeros((len(seqs), 4, seq_len), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for pos, base in enumerate(seq[:seq_len]):
            ch = BASE_TO_CHAN.get(base)
            if ch is not None:
                arr[i, ch, pos] = 1.0
    return arr


class DNADataset(Dataset):
    """One-hot [4, L] dataset fed directly to SimplexDenoiser."""

    def __init__(self, onehot: np.ndarray):
        self.onehot = torch.from_numpy(onehot)

    def __len__(self) -> int:
        return self.onehot.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.onehot[idx]


def load_clean_sequences(data_file: Path, seq_len: int) -> list[str]:
    """Read sequences from the first CSV column, keeping only pure-ACGT sequences of exactly seq_len."""
    df = pd.read_csv(data_file)
    col = "sequence" if "sequence" in df.columns else df.columns[0]
    valid: list[str] = []
    for seq in df[col].astype(str).tolist():
        seq = seq.upper().strip()
        if len(seq) == seq_len and set(seq) <= set("ACGT"):
            valid.append(seq)
    return valid


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.wait = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - 1e-6:
            self.best = val_loss
            self.wait = 0
            return True  # improved
        self.wait += 1
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--biopart", required=True, choices=list(BIOPART_DATA))
    p.add_argument("--data_path", default=None, help="path to the biopart training CSV (required)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--val_size", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bp = BIOPART_DATA[args.biopart]
    seq_len = bp["seq_len"]

    if not args.data_path:
        raise SystemExit("--data_path is required (no default dataset path is bundled)")
    data_file = Path(args.data_path)
    sequences = load_clean_sequences(data_file, seq_len)
    print(f"[{args.biopart}] loaded {len(sequences)} clean sequences ({seq_len} bp) from {data_file}")
    if len(sequences) < 1000:
        print(f"[Warning] Few sequences ({len(sequences)}); please verify the data path.")

    # Split follows the benchmark protocol: first 90% train, last 10% validation (no shuffling).
    n = len(sequences)
    n_train = int(n * (1 - args.val_size))
    train_arr = encode_onehot(sequences[:n_train], seq_len)
    val_arr = encode_onehot(sequences[n_train:], seq_len)
    train_loader = DataLoader(DNADataset(train_arr), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(DNADataset(val_arr), batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"[{args.biopart}] train {len(train_arr)} / val {len(val_arr)}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "diffusion_checkpoints" / f"{args.biopart}_direct"
    )
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Model and scheduler (default hyperparameters = defaults of get_direct_denoiser_model so the loader can restore them).
    model = get_direct_denoiser_model(seq_len=seq_len).to(device)
    schedule = VPNoiseSchedule(args.num_timesteps, args.beta_start, args.beta_end).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.biopart}] SimplexDenoiser parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early = EarlyStopping(args.patience)

    # Pre-collect constants for t sampling.
    T = args.num_timesteps

    def run_epoch(loader, train: bool) -> float:
        model.train() if train else model.eval()
        total, count = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x0 in loader:  # [B, 4, L]
                x0 = x0.to(device)
                b = x0.shape[0]
                t = torch.randint(0, T, (b,), device=device, dtype=torch.long)
                noise = torch.randn_like(x0)
                x_t = schedule.q_sample(x0, t, noise)
                noise_pred = model(x_t, t)
                loss = F.mse_loss(noise_pred, noise)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total += loss.item() * b
                count += b
        return total / max(count, 1)

    best_val = float("inf")
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"train {args.biopart}"):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        scheduler.step()

        improved = early.step(val_loss)
        if improved and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                output_dir / "checkpoints" / "best.pth",
            )
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{args.biopart}] epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}  best {best_val:.5f}")

        if early.wait >= args.patience:
            print(f"[{args.biopart}] early stopping @ epoch {epoch} (patience={args.patience})")
            break

    # Always persist a latest checkpoint.
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
        output_dir / "checkpoints" / "latest.pth",
    )

    # config.json: the loader reads num_timesteps / beta_start / beta_end to rebuild the scheduler.
    config = {
        "biopart": args.biopart,
        "model_type": "direct",
        "seq_len": seq_len,
        "num_timesteps": args.num_timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "epochs_run": epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "n_train": len(train_arr),
        "n_val": len(val_arr),
        "best_val_loss": best_val,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{args.biopart}] done. best val {best_val:.5f} -> {output_dir / 'checkpoints' / 'best.pth'}")


if __name__ == "__main__":
    main()
