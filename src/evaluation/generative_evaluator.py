#!/usr/bin/env python3
"""
Pure-computation evaluator for generative DNA sequence models.

Extracts all metric computation, sampling, and model-loading logic from
``diffusion_model/evaluate_diffusion.py`` while keeping visualization
concerns out of this module entirely.  Every public function returns
structured data (dicts, arrays) that downstream consumers can pass to
their visualization layer.

Functions that already live in ``core.metrics_kmer`` and
``core.metrics_distance`` are imported, not duplicated.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Path setup so that ``codes/``-relative imports work regardless of cwd.
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Consolidated metric helpers from the metrics/ package
from metrics import (
    compute_kmer_frequencies,
    kmer_correlation,
    kmer_js_divergence,
    compute_sequence_diversity,
    levenshtein_distance,
    compute_min_distances_to_reference as compute_min_distances_to_train,
    compute_distance_statistics,
)

# Diffusion model components — prefer new location, fall back to legacy
from models.diffusion import (
    SimplexDenoiser,
    VPNoiseSchedule,
    get_direct_denoiser_model,
)

# seq2onehot helper – try the canonical location first
try:
    from data.encoding import seq2onehot  # type: ignore[no-redef]
except ImportError:
    try:
        from utils.data import seq2onehot  # type: ignore[no-redef]
    except ImportError:
        def seq2onehot(sequences: List[str]) -> List[np.ndarray]:
            """Convert DNA sequences to one-hot encoding (TCGA order).

            Fallback implementation used when ``utils.data`` is not on the path.
            """
            # TCGA order: T=0, C=1, G=2, A=3
            module = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            onehot: List[np.ndarray] = []
            for seq in sequences:
                arr = np.zeros((len(seq), 4), dtype=np.float32)
                for i, nuc in enumerate(seq.upper()):
                    if nuc == 'A':
                        arr[i] = module[3]
                    elif nuc == 'C':
                        arr[i] = module[1]
                    elif nuc == 'G':
                        arr[i] = module[2]
                    elif nuc in ('T', 'U'):
                        arr[i] = module[0]
                onehot.append(arr)
            return onehot


logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class DNADataset(Dataset):
    """Thin ``torch.utils.data.Dataset`` wrapper around one-hot DNA sequences.

    Args:
        sequences: List of DNA sequence strings (all same length expected).
        seq_len: Expected sequence length (used for validation).
    """

    def __init__(self, sequences: List[str], seq_len: int) -> None:
        self.sequences = sequences
        self.seq_len = seq_len
        self.onehot = seq2onehot(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.onehot[idx], dtype=torch.float32)


# =============================================================================
# Sequence embedding and interpolation
# =============================================================================

def compute_sequence_embeddings(
    sequences: List[str],
    k: int = 3,
) -> np.ndarray:
    """Compute k-mer frequency embeddings for a list of DNA sequences.

    Each sequence is represented as a 4^k dimensional vector of normalized
    k-mer frequencies.  This is useful for downstream dimensionality
    reduction (t-SNE, UMAP) and similarity analyses.

    Args:
        sequences: List of DNA sequence strings.
        k: K-mer size.  Default 3 yields 64-dimensional embeddings.

    Returns:
        ``np.ndarray`` of shape ``(len(sequences), 4**k)``.
    """
    nucleotides = ["A", "C", "G", "T"]
    all_kmers = [
        "".join(p)
        for p in np.array(np.meshgrid(*[nucleotides] * k)).T.reshape(-1, k)
    ]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}

    embeddings: List[np.ndarray] = []
    for seq in sequences:
        counts = np.zeros(4**k, dtype=np.float32)
        total = 0
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if "N" not in kmer and kmer in kmer_to_idx:
                counts[kmer_to_idx[kmer]] += 1
                total += 1
        embeddings.append(counts / total if total > 0 else counts)

    return np.array(embeddings)


def interpolate_sequences(
    seq1: str,
    seq2: str,
    num_steps: int,
) -> List[str]:
    """Linearly interpolate between two DNA sequences in one-hot space.

    At each step a soft blend of the two one-hot vectors is computed and the
    argmax is taken to produce a discrete sequence.

    Args:
        seq1: First DNA sequence.
        seq2: Second DNA sequence (must be same length as *seq1*).
        num_steps: Number of interpolation steps (including endpoints).

    Returns:
        List of ``num_steps`` interpolated DNA sequence strings.
    """
    # TCGA order, consistent with seq2onehot()
    nucleotides = ["T", "C", "G", "A"]
    nuc_to_idx = {n: i for i, n in enumerate(nucleotides)}

    def _seq_to_onehot(seq: str) -> np.ndarray:
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, nuc in enumerate(seq.upper()):
            if nuc in nuc_to_idx:
                arr[i, nuc_to_idx[nuc]] = 1.0
            elif nuc == 'U':
                arr[i, nuc_to_idx['T']] = 1.0
        return arr

    onehot1 = _seq_to_onehot(seq1)
    onehot2 = _seq_to_onehot(seq2)

    interpolated: List[str] = []
    for t in np.linspace(0, 1, num_steps):
        interp = (1 - t) * onehot1 + t * onehot2
        seq = "".join(
            nucleotides[np.argmax(interp[i])] for i in range(len(interp))
        )
        interpolated.append(seq)
    return interpolated


# =============================================================================
# Diffusion sampling
# =============================================================================

def p_sample(
    denoiser: SimplexDenoiser,
    noise_schedule: VPNoiseSchedule,
    shape: Tuple[int, int, int],
    device: torch.device,
    num_inference_steps: int = 1000,
) -> torch.Tensor:
    """Run the reverse diffusion (denoising) process starting from pure noise.

    Args:
        denoiser: Trained ``SimplexDenoiser`` model.
        noise_schedule: ``VPNoiseSchedule`` providing noise parameters.
        shape: Desired output shape ``(B, 4, L)``.
        device: Torch device.
        num_inference_steps: Total number of denoising time-steps.

    Returns:
        Tensor of shape ``(B, 4, L)`` with denoised one-hot logits.
    """
    x = torch.randn(shape, device=device)

    for t in tqdm(reversed(range(num_inference_steps)), desc="Denoising"):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        with torch.no_grad():
            noise_pred = denoiser(x, t_tensor)

        alpha_t = noise_schedule.alphas[t].view(-1, 1, 1)
        alpha_t_cumprod = noise_schedule.alphas_cumprod[t].view(-1, 1, 1)
        beta_t = noise_schedule.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod[
            t
        ].view(-1, 1, 1)

        # Predicted x_0
        x_0_pred = (x - sqrt_one_minus_alpha_cumprod * noise_pred) / torch.sqrt(
            alpha_t_cumprod
        )

        # Posterior mean
        mean = (x - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod) / torch.sqrt(
            alpha_t
        )

        # Sample with posterior variance
        variance = noise_schedule.posterior_variance[t].view(-1, 1, 1)
        noise = torch.randn_like(x)
        nonzero_mask = 1.0 if t > 0 else 0.0
        x = mean + nonzero_mask * torch.sqrt(variance) * noise

    return x


def sample_sequences(
    denoiser: SimplexDenoiser,
    noise_schedule: VPNoiseSchedule,
    num_samples: int,
    seq_len: int,
    device: torch.device,
    num_inference_steps: int = 1000,
    temperature: float = 1.0,
) -> List[str]:
    """Generate DNA sequences from the direct diffusion model.

    Sequences are produced in batches to avoid GPU OOM.

    Args:
        denoiser: Trained ``SimplexDenoiser``.
        noise_schedule: ``VPNoiseSchedule`` for the diffusion process.
        num_samples: Number of sequences to generate.
        seq_len: Length of each generated sequence.
        device: Torch device.
        num_inference_steps: Denoising steps per sequence.
        temperature: Sampling temperature (reserved for future use).

    Returns:
        List of generated DNA sequence strings.
    """
    logger.info("Generating %d sequences with direct diffusion...", num_samples)
    denoiser.eval()

    batch_size = 1000
    nuc_mapping = {0: "T", 1: "C", 2: "G", 3: "A"}
    sequences: List[str] = []

    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        shape = (bs, 4, seq_len)
        x = p_sample(denoiser, noise_schedule, shape, device, num_inference_steps)

        for i in range(bs):
            x_seq = x[i].T  # [L, 4]
            indices = torch.argmax(x_seq, dim=-1).cpu().numpy()
            seq = "".join(nuc_mapping[idx] for idx in indices)
            sequences.append(seq)

        logger.info("Generated %d/%d sequences", len(sequences), num_samples)

    logger.info("Generated %d sequences total", len(sequences))
    return sequences


# =============================================================================
# Model loading
# =============================================================================

def load_direct_diffusion_model(
    checkpoint_dir: str,
    seq_len: int,
    device: torch.device,
) -> Tuple[SimplexDenoiser, VPNoiseSchedule, Dict[str, Any]]:
    """Load a trained direct diffusion model from a checkpoint directory.

    Expected directory layout::

        checkpoint_dir/
            config.json
            checkpoints/
                best.pth

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        seq_len: Sequence length the model was trained on.
        device: Torch device to load the model onto.

    Returns:
        ``(denoiser, noise_schedule, config_dict)`` tuple.

    Raises:
        FileNotFoundError: If the denoiser checkpoint file is missing.
    """
    checkpoint_path = Path(checkpoint_dir)

    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config: Dict[str, Any] = json.load(f)
    else:
        config = {}

    # Load denoiser weights
    denoiser_checkpoint = checkpoint_path / "checkpoints" / "best.pth"
    if not denoiser_checkpoint.exists():
        raise FileNotFoundError(
            f"Denoiser checkpoint not found: {denoiser_checkpoint}"
        )

    denoiser = get_direct_denoiser_model(seq_len)
    denoiser_state = torch.load(
        denoiser_checkpoint, map_location=device, weights_only=False
    )
    denoiser.load_state_dict(denoiser_state["model_state_dict"])
    denoiser = denoiser.to(device)
    denoiser.eval()
    logger.info("Loaded denoiser from %s", denoiser_checkpoint)

    # Create noise schedule from config
    noise_schedule = VPNoiseSchedule(
        num_timesteps=config.get("num_timesteps", 1000),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 2e-2),
    ).to(device)

    return denoiser, noise_schedule, config


# =============================================================================
# High-level evaluation orchestrator
# =============================================================================

def evaluate_generative_model(
    train_seqs: List[str],
    gen_seqs: List[str],
    *,
    kmer_k: int = 4,
    embedding_k: int = 3,
    tsne_perplexity: Optional[int] = None,
    num_interp_paths: int = 3,
    num_interp_steps: int = 10,
    interp_seed: int = 42,
) -> Dict[str, Any]:
    """Run all generative-quality evaluations and return structured results.

    This is the main entry point for headless / programmatic evaluation.
    It computes k-mer correlations, edit-distance novelty statistics, t-SNE
    embeddings, and interpolation smoothness metrics.  **No plots are
    created**; everything is returned as plain Python / NumPy objects.

    Args:
        train_seqs: Training (reference) sequences.
        gen_seqs: Generated (model output) sequences.
        kmer_k: K-mer size for frequency correlation analysis.
        embedding_k: K-mer size for t-SNE embeddings.
        tsne_perplexity: Perplexity for t-SNE.  ``None`` auto-selects
            ``min(30, n_samples // 4)``.
        num_interp_paths: Number of random interpolation paths.
        num_interp_steps: Steps per interpolation path.
        interp_seed: Random seed for selecting interpolation pairs.

    Returns:
        Dictionary with the following keys:

        * ``kmer_correlation_log`` – Pearson r / R^2 / p on log10 frequencies
          (non-zero only).
        * ``kmer_correlation_raw`` – Pearson r / R^2 / p on raw frequencies.
        * ``kmer_js_divergence`` – JS divergence between k-mer distributions.
        * ``min_edit_distances`` – ``np.ndarray`` of per-sequence min distances.
        * ``edit_distance_stats`` – Summary statistics dict.
        * ``sequence_diversity`` – Fraction of unique generated sequences.
        * ``tsne_train`` – ``np.ndarray`` of shape ``(n_train, 2)``.
        * ``tsne_gen`` – ``np.ndarray`` of shape ``(n_gen, 2)``.
        * ``interpolation_paths`` – List of dicts (one per path) with keys
          ``pair_indices``, ``sequences``, ``kmer_correlations``.
    """
    results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. K-mer frequency correlation (log-scale, non-zero only)
    # ------------------------------------------------------------------
    logger.info("[1/5] Computing k-mer frequency correlations (k=%d)...", kmer_k)
    train_freqs = compute_kmer_frequencies(train_seqs, k=kmer_k)
    gen_freqs = compute_kmer_frequencies(gen_seqs, k=kmer_k)

    kmers = sorted(train_freqs.keys())
    train_vals = np.array([train_freqs[k] for k in kmers])
    gen_vals = np.array([gen_freqs.get(k, 0.0) for k in kmers])

    # Log-scale correlation (non-zero only)
    mask = (train_vals > 0) & (gen_vals > 0)
    train_log = np.log10(train_vals[mask] + 1e-10)
    gen_log = np.log10(gen_vals[mask] + 1e-10)
    r_log, p_log = pearsonr(train_log, gen_log)
    results["kmer_correlation_log"] = {
        "pearson_r": float(r_log),
        "r_squared": float(r_log**2),
        "p_value": float(p_log),
    }
    logger.info(
        "  Log-scale: r=%.4f, R^2=%.4f, p=%.4e",
        r_log,
        r_log**2,
        p_log,
    )

    # Raw correlation (all k-mers)
    r_raw, p_raw = pearsonr(train_vals, gen_vals)
    results["kmer_correlation_raw"] = {
        "pearson_r": float(r_raw),
        "r_squared": float(r_raw**2),
        "p_value": float(p_raw),
    }
    logger.info(
        "  Raw: r=%.4f, R^2=%.4f, p=%.4e",
        r_raw,
        r_raw**2,
        p_raw,
    )

    # ------------------------------------------------------------------
    # 2. K-mer JS divergence
    # ------------------------------------------------------------------
    js_div = kmer_js_divergence(train_freqs, gen_freqs)
    results["kmer_js_divergence"] = float(js_div)
    logger.info("  JS divergence: %.6f", js_div)

    # ------------------------------------------------------------------
    # 3. Minimum edit distances (novelty)
    # ------------------------------------------------------------------
    logger.info("[2/5] Computing minimum edit distances to training set...")
    min_distances = compute_min_distances_to_train(gen_seqs, train_seqs)
    results["min_edit_distances"] = min_distances
    dist_stats = compute_distance_statistics(min_distances)
    results["edit_distance_stats"] = dist_stats
    logger.info(
        "  Mean: %.2f, Median: %.2f, Min: %.0f, Max: %.0f",
        dist_stats["mean"],
        dist_stats["median"],
        dist_stats["min"],
        dist_stats["max"],
    )

    # ------------------------------------------------------------------
    # 4. Sequence diversity
    # ------------------------------------------------------------------
    diversity = compute_sequence_diversity(gen_seqs)
    results["sequence_diversity"] = diversity
    logger.info("[3/5] Sequence diversity: %.4f", diversity)

    # ------------------------------------------------------------------
    # 5. t-SNE embeddings
    # ------------------------------------------------------------------
    logger.info("[4/5] Computing t-SNE embeddings...")
    train_emb = compute_sequence_embeddings(train_seqs, k=embedding_k)
    gen_emb = compute_sequence_embeddings(gen_seqs, k=embedding_k)
    combined = np.vstack([train_emb, gen_emb])

    perp = tsne_perplexity or min(30, len(combined) // 4)
    reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
    embedding_2d = reducer.fit_transform(combined)

    n_train = len(train_emb)
    results["tsne_train"] = embedding_2d[:n_train]
    results["tsne_gen"] = embedding_2d[n_train:]
    logger.info("  t-SNE complete: %d train + %d generated points", n_train, len(gen_emb))

    # ------------------------------------------------------------------
    # 6. Interpolation smoothness
    # ------------------------------------------------------------------
    logger.info("[5/5] Computing interpolation smoothness (%d paths)...", num_interp_paths)
    rng = np.random.RandomState(interp_seed)
    train_kmer_freq = compute_kmer_frequencies(train_seqs, k=kmer_k)
    interp_paths: List[Dict[str, Any]] = []

    for path_idx in range(num_interp_paths):
        idx1, idx2 = rng.choice(len(gen_seqs), 2, replace=False)
        seq1, seq2 = gen_seqs[idx1], gen_seqs[idx2]

        interpolated = interpolate_sequences(seq1, seq2, num_interp_steps)
        correlations: List[float] = []

        for seq in interpolated:
            seq_kmer_freq = compute_kmer_frequencies([seq], k=kmer_k)
            common_kmers = list(set(seq_kmer_freq.keys()) & set(train_kmer_freq.keys()))
            if len(common_kmers) > 1:
                sv = np.array([seq_kmer_freq[k] for k in common_kmers])
                tv = np.array([train_kmer_freq[k] for k in common_kmers])
                corr, _ = pearsonr(sv, tv)
                correlations.append(float(corr) if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)

        interp_paths.append(
            {
                "pair_indices": (int(idx1), int(idx2)),
                "sequences": interpolated,
                "kmer_correlations": correlations,
            }
        )

    results["interpolation_paths"] = interp_paths
    logger.info("  Interpolation analysis complete")

    return results
