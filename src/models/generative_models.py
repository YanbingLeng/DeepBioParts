#!/usr/bin/env python3
"""Unified module for DeepBioParts deep generative model architectures.

This module gathers the three deep generative model architectures used for the
fair comparison in Fig. 3b:

* :class:`DNAVAE` -- a self-contained DNA variational autoencoder (VAE) with
  its own Encoder/Decoder.
* :class:`DNAGAN` -- a DNA sequence GAN (generator + discriminator).
* :class:`SimplexDenoiser` -- a DDPM operating directly in one-hot sequence
  space. Its source definition lives in ``src/models/diffusion/denoiser.py``
  and is reused by the production pipeline (the default generative model);
  it is merely re-exported here so that this module is the single entry point
  for all three architectures.

Design constraints
------------------
Trained checkpoints exist for all three architectures
(``fair_checkpoints/{bp}_{vae,gan,ddpm}_fair/`` and
``diffusion_checkpoints/{bp}_direct/``). So that these classes can load
the existing weights without retraining, the internal sub-networks
(``_VAEEncoder`` / ``_VAEDecoder`` / ``_Generator`` / ``_Discriminator``)
replicate the original implementations exactly -- layer names and tensor
shapes are identical, so the ``state_dict`` keys are unchanged.

Import convention: ``from models.generative_models import DNAVAE, DNAGAN, SimplexDenoiser``
(consistent with ``models.diffusion.*``; requires ``src`` on ``sys.path``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# DDPM architecture (source definition in diffusion/denoiser.py) and noise
# schedule: re-exported, not reimplemented.
from .diffusion.denoiser import (  # noqa: F401
    SimplexDenoiser,
    get_direct_denoiser_model,
)
from .diffusion.noise_schedule import VPNoiseSchedule  # noqa: F401

__all__ = [
    "DNAVAE",
    "DNAGAN",
    "SimplexDenoiser",
    "get_direct_denoiser_model",
    "VPNoiseSchedule",
]


# ---------------------------------------------------------------------------
# VAE: self-contained DNA variational autoencoder
# ---------------------------------------------------------------------------

class _VAEEncoder(nn.Module):
    """Map one-hot [B, 4, L] to a latent vector [B, latent_dim].

    Flatten + MLP (empirically effective for short DNA sequences).
    ``state_dict`` keys: ``net.0/2``, ``fc_mu``, ``fc_logvar``.
    """

    def __init__(self, seq_len: int = 40, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.seq_len = seq_len
        input_dim = seq_len * 4  # flattened one-hot

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        h = x.permute(0, 2, 1).reshape(b, -1)  # [B, 4*L]
        h = self.net(h)
        return self.fc_mu(h), self.fc_logvar(h)


class _VAEDecoder(nn.Module):
    """Map a latent vector [B, latent_dim] to one-hot logits [B, 4, L].

    Symmetric MLP; ``state_dict`` keys: ``net.0/2/4``.
    """

    def __init__(self, seq_len: int = 40, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.seq_len = seq_len
        output_dim = seq_len * 4

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)                       # [B, seq_len * 4]
        return h.view(-1, 4, self.seq_len)    # [B, 4, seq_len]


class DNAVAE(nn.Module):
    """Self-contained DNA variational autoencoder (VAE).

    ``self.encoder`` / ``self.decoder`` replicate the original sub-networks, so
    the ``encoder_state_dict`` / ``decoder_state_dict`` of existing checkpoints
    can be loaded directly, without retraining.

    Args:
        seq_len: Sequence length.
        latent_dim: Latent space dimension.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(self, seq_len: int = 40, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = _VAEEncoder(seq_len, latent_dim, hidden_dim)
        self.decoder = _VAEDecoder(seq_len, latent_dim, hidden_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map [B, 4, L] to (mu, logvar), each [B, latent_dim]."""
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map [B, latent_dim] to reconstruction logits [B, 4, L]."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction: x -> (mu, logvar) -> z -> logits."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def sample_prior(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample z ~ N(0, I) from the prior and decode; returns logits [n, 4, L]."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


# ---------------------------------------------------------------------------
# GAN: DNA sequence generative adversarial network (generator + discriminator)
# ---------------------------------------------------------------------------

class _Generator(nn.Module):
    """MLP generator: latent -> [B, 4, L] logits.

    ``state_dict`` keys: ``net.0/1/3/4/6``.
    """

    def __init__(self, seq_len: int, latent_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 4 * seq_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 4, self.seq_len)


class _Discriminator(nn.Module):
    """Conv1d discriminator (spectral norm for training stability)."""

    def __init__(self, seq_len: int, base: int = 64):
        super().__init__()
        from torch.nn.utils import spectral_norm as sn
        self.net = nn.Sequential(
            sn(nn.Conv1d(4, base, 3, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv1d(base, base * 2, 3, stride=2, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv1d(base * 2, base * 4, 3, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            sn(nn.Linear(base * 4, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DNAGAN(nn.Module):
    """DNA sequence GAN wrapping a generator and a discriminator.

    ``self.generator`` replicates the original architecture, so the
    ``generator_state_dict`` of existing checkpoints loads directly.
    Gumbel-softmax provides differentiable sampling in one-hot space; decoding
    takes the argmax over the 4 channels in TCGA order (T=0, C=1, G=2, A=3).

    Args:
        seq_len: Sequence length.
        latent_dim: Dimension of the generator's latent noise.
        hidden: MLP hidden dimension.
    """

    def __init__(self, seq_len: int, latent_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.generator = _Generator(seq_len, latent_dim, hidden)
        self.discriminator = _Discriminator(seq_len)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate: z -> logits [B, 4, L] (equivalent to ``self.generator(z)``)."""
        return self.generator(z)
