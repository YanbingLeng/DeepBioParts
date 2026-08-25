#!/usr/bin/env python3
"""
Neural network components for the direct diffusion denoiser.

This module contains the building blocks and the main denoiser architecture
used for continuous diffusion on one-hot encoded DNA sequences (the probability
simplex).  The architecture is UNet-like with sinusoidal time embeddings,
residual blocks, and self-attention.

Classes:
    SinusoidalPositionEmbeddings - Sinusoidal time-step embeddings.
    AttentionBlock - Self-attention block with residual connection.
    ResidualBlock - Residual block with FiLM-style time conditioning.
    SimplexDenoiser - Full denoiser network (UNet-style).

Functions:
    get_direct_denoiser_model - Factory that creates a ``SimplexDenoiser``.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps.

    Maps scalar timesteps to a ``dim``-dimensional vector using the
    sinusoidal encoding scheme from *Attention Is All You Need*.

    Args:
        dim: Embedding dimensionality (output size).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings.

        Args:
            time: Timestep values, shape ``[B]``.

        Returns:
            Embedding tensor of shape ``[B, dim]``.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """Self-attention block for 1-D sequence feature maps.

    Uses grouped normalization, multi-head self-attention, and a residual
    connection.  The implementation is based on ``nn.Conv1d`` projections.

    Args:
        channels: Number of input (and output) channels.
        num_heads: Number of attention heads.  ``channels`` must be divisible
            by this value.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.norm = nn.GroupNorm(1, channels)

        # Use separate projections for better control
        self.to_qkv = nn.Conv1d(channels, channels * 3, 1)
        self.to_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run self-attention with a residual connection.

        Args:
            x: Input feature map, shape ``[B, C, L]``.

        Returns:
            Output feature map, same shape as *x*.
        """
        B, C, L = x.shape
        x_norm = self.norm(x)

        qkv = self.to_qkv(x_norm)  # [B, 3*C, L]
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv.unbind(1)  # [B, heads, C//heads, L]

        # Attention
        scale = (C // self.num_heads) ** -0.5
        # Reshape for attention: [B, heads, L, C//heads]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, L, C//heads]
        out = out.transpose(-2, -1)  # [B, heads, C//heads, L]
        out = out.reshape(B, C, L)

        return x + self.to_out(out)


class ResidualBlock(nn.Module):
    """Residual block with FiLM-style time-step conditioning.

    Two ``GroupNorm`` + ``Conv1d`` stages with SiLU activations.  The
    intermediate features are modulated (scale + shift) by a learned
    projection of the time embedding.

    Args:
        channels: Number of input / output channels.
        emb_dim: Dimensionality of the time embedding.
        dropout: Dropout rate applied between the two convolutions.
    """

    def __init__(self, channels: int, emb_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

        # Time embedding projection
        self.time_emb = nn.Linear(emb_dim, channels * 2)

        self.dropout = nn.Dropout(dropout)
        self.residual_conv = nn.Conv1d(channels, channels, 1) if channels != 4 else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature map, shape ``[B, C, L]``.
            time_emb: Time embedding, shape ``[B, emb_dim]``.

        Returns:
            Output feature map, same shape as *x*.
        """
        B, C, L = x.shape

        # Time conditioning
        t = self.time_emb(time_emb)  # [B, 2*C]
        t_scale, t_shift = t.chunk(2, dim=1)  # [B, C] each
        t_scale = t_scale[:, :, None]
        t_shift = t_shift[:, :, None]

        # First conv
        h = self.norm1(x)
        h = F.silu(h)
        h = h * t_scale + t_shift  # Modulation
        h = self.conv1(h)

        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return x + h


class SimplexDenoiser(nn.Module):
    """Direct denoiser for one-hot encoded DNA sequences.

    Operates on the probability simplex (4 channels per position).
    Uses a UNet-like architecture with attention.

    The network downsamples the input sequence by a factor of 4, processes
    it with residual and attention blocks at the coarsest resolution, then
    upsamples back while aggregating skip connections.

    Args:
        seq_len: Expected sequence length (used for adaptive pooling).
        time_emb_dim: Dimensionality of the time embedding.
        channels: Base channel count (multiplied at deeper levels).
        num_residual_blocks: Number of residual blocks at the bottleneck.
        num_heads: Number of attention heads in the bottleneck attention.
    """

    def __init__(
        self,
        seq_len: int = 40,
        time_emb_dim: int = 256,
        channels: int = 128,
        num_residual_blocks: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Input projection (4 channels -> channels)
        self.input_proj = nn.Conv1d(4, channels, 3, padding=1)

        # Downsampling (reduce sequence length)
        self.down1 = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels * 2, 3, stride=2, padding=1),  # L -> L/2
        )

        self.down2 = nn.Sequential(
            nn.GroupNorm(1, channels * 2),
            nn.SiLU(),
            nn.Conv1d(channels * 2, channels * 4, 3, stride=2, padding=1),  # L/2 -> L/4
        )

        # Middle blocks (at lowest resolution)
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(channels * 4, time_emb_dim)
            for _ in range(num_residual_blocks)
        ])
        self.mid_attn = AttentionBlock(channels * 4, num_heads)

        # Upsampling
        self.up1 = nn.Sequential(
            nn.GroupNorm(1, channels * 4),
            nn.SiLU(),
            nn.ConvTranspose1d(channels * 4, channels * 2, 4, stride=2, padding=1),  # L/4 -> L/2
        )

        self.up2 = nn.Sequential(
            nn.GroupNorm(1, channels * 2),
            nn.SiLU(),
            nn.ConvTranspose1d(channels * 2, channels, 4, stride=2, padding=1),  # L/2 -> L
        )

        # Output projection (channels -> 4 channels)
        self.output_norm = nn.GroupNorm(1, channels)
        self.output_conv = nn.Conv1d(channels, 4, 3, padding=1)

        # Adaptive pooling to ensure output matches input sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(seq_len)

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Predict noise given a noisy input and a diffusion timestep.

        Args:
            x_t: Noisy one-hot encoded sequences, shape ``[B, 4, L]``.
            timestep: Diffusion timesteps, shape ``[B]``.

        Returns:
            Predicted noise, shape ``[B, 4, seq_len]``.
        """
        # Time embedding
        time_emb = self.time_mlp(timestep)  # [B, time_emb_dim]

        # Input projection
        h = self.input_proj(x_t)  # [B, channels, L]

        # Downsampling with skip connections
        h1 = self.down1(h)  # [B, channels*2, L/2]
        h2 = self.down2(h1)  # [B, channels*4, L/4]

        # Middle processing
        h = h2
        for block in self.mid_blocks:
            h = block(h, time_emb)
        h = self.mid_attn(h) + h  # [B, channels*4, L/4]

        # Upsampling with skip connections (concatenate instead of add)
        h = self.up1(h)  # [B, channels*2, L/2]
        # Simple skip connection (crop if needed)
        if h.shape[2] != h1.shape[2]:
            min_len = min(h.shape[2], h1.shape[2])
            h = h[:, :, :min_len] + h1[:, :, :min_len]
        else:
            h = h + h1

        h = self.up2(h)  # [B, channels, L]
        # Simple skip connection
        if h.shape[2] != h.shape[2]:
            min_len = min(h.shape[2], h.shape[2])
            h = h[:, :, :min_len] + h[:, :, :min_len]
        else:
            h = h + h

        # Output
        h = self.output_norm(h)
        h = F.silu(h)

        # Ensure output matches expected sequence length
        h = self.adaptive_pool(h)  # [B, channels, seq_len]

        noise_pred = self.output_conv(h)  # [B, 4, seq_len]

        return noise_pred


def get_direct_denoiser_model(seq_len: int = 40) -> SimplexDenoiser:
    """Factory function to create a ``SimplexDenoiser`` with default hyper-parameters.

    Args:
        seq_len: Length of the DNA sequences to be denoised.

    Returns:
        A newly instantiated ``SimplexDenoiser``.
    """
    return SimplexDenoiser(
        seq_len=seq_len,
        time_emb_dim=256,
        channels=128,
        num_residual_blocks=3,
        num_heads=4,
    )
