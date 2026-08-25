"""Unified LoRA (Low-Rank Adaptation) implementation.

``LoRALinear`` used by the Evo fine-tuning pipeline, supporting two
usage patterns:

* **Reference mode:** Call ``set_original_weight(weight, bias)``
  once, then call ``forward(x)`` with no extra arguments.  The frozen weight
  is read by reference (no copy).
* **External-base mode:** Omit ``set_original_weight`` and pass
  ``base_weight`` directly to ``forward(x, base_weight)``.
* **Merged mode:** Call ``merge_weights(base_weight)`` to obtain a single
  weight tensor with LoRA baked in (useful for inference).

Args:
    in_features: Dimensionality of the input.
    out_features: Dimensionality of the output.
    rank: LoRA rank (number of low-rank dimensions).  Typical values: 4--64.
    alpha: LoRA scaling factor.  The effective scaling is ``alpha / rank``.
    dropout: Dropout probability applied *before* the LoRA projection.
    dtype: Optional ``torch.dtype`` for the LoRA parameter tensors.  When
        ``None`` (default) the parameters use PyTorch's default dtype.

Example::

    # Evo-style usage
    lora = LoRALinear(4096, 4096, rank=16, alpha=32, dtype=torch.bfloat16)
    lora.set_original_weight(linear.weight, linear.bias)
    out = lora(x)

    # External-base usage
    lora = LoRALinear(128, 64, rank=8, alpha=16)
    out = lora(x, base_weight=linear.weight)
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Unified Low-Rank Adaptation layer for ``nn.Linear`` modules.

    Implements the LoRA mechanism: ``h = Wx + (alpha/rank) * B @ A @ x``,
    where *W* is the frozen pre-trained weight and *A*, *B* are learnable
    low-rank matrices.

    Two usage variants are supported:

    * *Reference-weight* mode, which holds a pointer to the original weight
      and uses a configurable dtype (as used in ``evo.lora_finetune_evo``).
    * *External-base-weight* mode, which expects the caller to supply the
      base weight on every forward pass.

    Attributes:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        scaling: Effective scaling (``alpha / rank``).
        lora_A: Learnable low-rank matrix *A* of shape ``(rank, in_features)``.
        lora_B: Learnable low-rank matrix *B* of shape ``(out_features, rank)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the LoRA layer.

        Args:
            in_features: Dimensionality of the input.
            out_features: Dimensionality of the output.
            rank: LoRA rank (number of low-rank dimensions).
            alpha: LoRA scaling factor.  The effective scaling is ``alpha / rank``.
            dropout: Dropout probability applied before the LoRA projection.
            dtype: Optional ``torch.dtype`` for LoRA parameters.  When ``None``
                the default PyTorch dtype is used.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling: float = alpha / rank
        self.dtype = dtype

        # LoRA low-rank matrices.
        # A is initialised with small random values; B starts at zero so
        # that the LoRA contribution is zero at initialisation.
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=dtype)
        )

        # Dropout applied to the input before the LoRA path.
        self.lora_dropout = nn.Dropout(p=dropout)

        # ---- Reference-weight mode (Evo-style) ----
        # These are *not* ``nn.Parameters`` -- they are plain tensors (or
        # ``None``) stored as buffers so they move with ``.to(device)`` but
        # are never updated by the optimiser.
        self._original_weight: Optional[torch.Tensor] = None
        self._original_bias: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Reference-weight helpers (Evo-style)
    # ------------------------------------------------------------------

    def set_original_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Store a *reference* to the original frozen weight (no copy).

        This is the Evo-style usage: call once after construction so that
        ``forward(x)`` can compute the full output without an externally
        supplied base weight.

        Args:
            weight: The frozen pre-trained weight tensor of shape
                ``(out_features, in_features)``.
            bias: Optional frozen bias tensor of shape ``(out_features,)``.
        """
        self._original_weight = weight
        self._original_bias = bias

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        base_weight: Optional[torch.Tensor] = None,
        base_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ``Wx + (alpha/rank) * B @ A @ dropout(x)``.

        The frozen base weight can be provided in one of two ways:

        1. **Externally** via the ``base_weight`` argument (DeepBioParts-style).
        2. **Internally** via ``set_original_weight`` (Evo-style).  In this
           case ``base_weight`` should be ``None``.

        Args:
            x: Input tensor of shape ``(..., in_features)``.
            base_weight: Optional frozen weight tensor of shape
                ``(out_features, in_features)``.  When ``None`` the weight
                set via ``set_original_weight`` is used.
            base_bias: Optional frozen bias tensor.  When ``None`` and a
                reference bias was set, the reference bias is used.

        Returns:
            Output tensor of shape ``(..., out_features)``.

        Raises:
            RuntimeError: If neither ``base_weight`` nor a reference weight
                has been set.
        """
        # Resolve the frozen weight.
        w = base_weight if base_weight is not None else self._original_weight
        b = base_bias if base_bias is not None else self._original_bias

        if w is None:
            raise RuntimeError(
                "LoRALinear requires a base weight.  Either pass "
                "`base_weight` to forward() or call set_original_weight() "
                "first."
            )

        # Standard linear transformation with frozen weights.
        result = F.linear(x, w, b)

        # LoRA path: dropout -> A^T -> B^T  (scaled).
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)  # (..., rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (..., out_features)

        return result + self.scaling * lora_out

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge the LoRA adaptation into the base weight.

        Returns a new tensor ``W_merged = W + (alpha/rank) * B @ A``
        without modifying any parameters.

        Args:
            base_weight: The frozen weight tensor of shape
                ``(out_features, in_features)``.

        Returns:
            Merged weight tensor of the same shape as ``base_weight``.
        """
        with torch.no_grad():
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            return base_weight + lora_weight

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Return only the LoRA parameters ``[lora_A, lora_B]``.

        Useful for constructing an optimiser that trains *only* the LoRA
        matrices while keeping the base model frozen.

        Returns:
            A list containing ``self.lora_A`` and ``self.lora_B``.
        """
        return [self.lora_A, self.lora_B]


def apply_lora_to_linear(
    linear_module: nn.Linear,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
) -> LoRALinear:
    """Replace an ``nn.Linear`` module with a LoRA-enabled version.

    Convenience factory that constructs a ``LoRALinear`` matching the
    dimensions of *linear_module*, copies the dtype from the original weight,
    and stores a reference to the original weight via
    ``set_original_weight``.

    Args:
        linear_module: The ``nn.Linear`` layer to wrap.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout probability for the LoRA path.

    Returns:
        A ``LoRALinear`` instance that is ready to use as a drop-in
        replacement for *linear_module*.
    """
    dtype = linear_module.weight.dtype
    lora_linear = LoRALinear(
        in_features=linear_module.in_features,
        out_features=linear_module.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        dtype=dtype,
    )
    lora_linear.set_original_weight(linear_module.weight, linear_module.bias)
    return lora_linear
