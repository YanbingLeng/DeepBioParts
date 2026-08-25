#!/usr/bin/env python3
"""
Noise schedule for variance-preserving (VP) continuous diffusion on the simplex.

This module provides the ``VPNoiseSchedule`` class which implements the forward
diffusion process (q-sample) and stores all derived quantities needed for
training and sampling (cumulative products of alphas, posterior variances, etc.).
"""

import torch
import torch.nn.functional as F


class VPNoiseSchedule:
    """Variance Preserving (VP) noise schedule for continuous diffusion on simplex.

    Uses a linear beta schedule, analogous to DDPM, but adapted for continuous
    data on the probability simplex.

    Args:
        num_timesteps: Total number of diffusion timesteps ``T``.
        beta_start: Initial value of the beta schedule.
        beta_end: Final value of the beta schedule.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Posterior variance
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

    def to(self, device: torch.device) -> "VPNoiseSchedule":
        """Move all pre-computed tensors to the given device.

        Args:
            device: Target device (e.g. ``torch.device("cuda:0")``).

        Returns:
            ``self``, for chaining.
        """
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward diffusion step: add noise to clean data.

        Computes ``sqrt(bar{alpha}_t) * x_0 + sqrt(1 - bar{alpha}_t) * noise``.

        Args:
            x_0: Clean data tensor, shape ``[B, C, L]`` or ``[B, D]`` (latent).
            t: Timestep indices (0-based), shape ``[B]``.
            noise: Optional pre-generated noise of the same shape as *x_0*.
                If ``None``, standard Gaussian noise is sampled.

        Returns:
            Noisy data tensor, same shape as *x_0*.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Auto-detect dimensionality for proper broadcasting:
        #   3D [B, C, L] (sequence data)  -> view as [B, 1, 1]
        #   2D [B, D]     (latent vectors) -> view as [B, 1]
        if x_0.ndim == 3:
            shape = (-1, 1, 1)
        elif x_0.ndim == 2:
            shape = (-1, 1)
        else:
            shape = (-1,) + (1,) * (x_0.ndim - 1)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(shape)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
