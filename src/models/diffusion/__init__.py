#!/usr/bin/env python3
"""
``models.diffusion`` -- neural-network components and noise schedules for
direct diffusion on one-hot encoded DNA sequences.

This package re-exports the public API so that downstream code can write::

    from models.diffusion import SimplexDenoiser, VPNoiseSchedule

Re-exported names
-----------------
.. autoclass:: SimplexDenoiser
.. autoclass:: SinusoidalPositionEmbeddings
.. autoclass:: AttentionBlock
.. autoclass:: ResidualBlock
.. autoclass:: VPNoiseSchedule
.. autofunction:: get_direct_denoiser_model
"""

from __future__ import annotations

# Neural-network denoiser components
from models.diffusion.denoiser import (
    SinusoidalPositionEmbeddings,
    AttentionBlock,
    ResidualBlock,
    SimplexDenoiser,
    get_direct_denoiser_model,
)

# Noise schedule
from models.diffusion.noise_schedule import VPNoiseSchedule

__all__ = [
    # Denoiser building blocks
    "SinusoidalPositionEmbeddings",
    "AttentionBlock",
    "ResidualBlock",
    "SimplexDenoiser",
    "get_direct_denoiser_model",
    # Noise schedule
    "VPNoiseSchedule",
]
