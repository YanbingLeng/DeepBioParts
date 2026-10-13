"""Unified model facade for backward-compatible imports.

This package re-exports key model classes from their original locations so that
downstream code can import from a single ``models`` package without depending
on internal module layout details.

Example::

    from models import ModelFactory, LoRALinear

Each import block is wrapped in ``try/except`` so the package degrades
gracefully when a sub-module is unavailable (e.g., missing dependencies or
partial installations).

Attributes:
    ModelFactory: Factory class for registering and creating predictor models.
    ConvPredictor: Attention-augmented bidirectional LSTM predictor model.
    OneDCNN: Lightweight 1D-CNN predictor for short sequences.
    ORNet: Ordinal regression output layer.
    BidirectionalLSTM: Bidirectional LSTM module.
    DenseBlock: Dense connectivity block (DenseNet-style).
    BottleneckLayer: Bottleneck layer inside DenseBlock.
    TransitionLayer: Transition layer between DenseBlocks.
    LoRALinear: Low-rank adaptation wrapper for nn.Linear.
    SimplexDenoiser: Direct denoiser for one-hot encoded DNA sequences.
    VPNoiseSchedule: Variance-preserving noise schedule for continuous diffusion.
    get_direct_denoiser_model: Factory function that creates a SimplexDenoiser.
"""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Naive supervised predictor models
# ---------------------------------------------------------------------------
try:
    from models.predictor import (
        ModelFactory,
        ConvPredictor,
        OneDCNN,
    )
except ImportError:
    ModelFactory = None  # type: ignore[assignment,misc]
    ConvPredictor = None  # type: ignore[assignment,misc]
    OneDCNN = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Utility network modules (ORNet, BiLSTM, DenseNet-style blocks)
# ---------------------------------------------------------------------------
try:
    from utils.module import (
        ORNet,
        BidirectionalLSTM,
        DenseBlock,
        BottleneckLayer,
        TransitionLayer,
    )
except ImportError:
    ORNet = None  # type: ignore[assignment,misc]
    BidirectionalLSTM = None  # type: ignore[assignment,misc]
    DenseBlock = None  # type: ignore[assignment,misc]
    BottleneckLayer = None  # type: ignore[assignment,misc]
    TransitionLayer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# LoRA adaptation
# ---------------------------------------------------------------------------
try:
    from models.lora import LoRALinear
except ImportError:
    LoRALinear = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Diffusion model: direct diffusion on simplex
# ---------------------------------------------------------------------------
try:
    from models.diffusion import (
        SimplexDenoiser,
        VPNoiseSchedule,
        get_direct_denoiser_model,
    )
except ImportError:
    SimplexDenoiser = None  # type: ignore[assignment,misc]
    VPNoiseSchedule = None  # type: ignore[assignment,misc]
    get_direct_denoiser_model = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Public ``__all__`` listing
# ---------------------------------------------------------------------------
__all__: List[str] = [
    # Naive supervised models
    "ModelFactory",
    "ConvPredictor",
    "OneDCNN",
    # Utility modules
    "ORNet",
    "BidirectionalLSTM",
    "DenseBlock",
    "BottleneckLayer",
    "TransitionLayer",
    # LoRA
    "LoRALinear",
    # Diffusion - direct
    "SimplexDenoiser",
    "VPNoiseSchedule",
    "get_direct_denoiser_model",
]
