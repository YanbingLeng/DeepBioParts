"""Loss functions for model training.

This module provides loss functions used across the DeepBioParts training
pipeline.  The mathematical logic is identical to the original implementations
in ``training.predictor``.

Losses:
    regression_loss: MSE + L1 combined loss for regression tasks.
    ordinal_regression_loss: Ordinal regression loss with auxiliary bias
        ordering penalty.
    classification_loss: Cross-entropy loss for classification tasks.

Example:
    >>> import torch
    >>> pred = torch.randn(8, 1)
    >>> target = torch.randn(8, 1)
    >>> loss = regression_loss(pred, target, l1_weight=0.1)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.1,
) -> torch.Tensor:
    """Combined MSE + L1 regression loss.

    Computes ``MSE(pred, target) + l1_weight * L1(pred, target)``.

    Args:
        pred: Predicted values of shape ``(batch_size, 1)`` or
            ``(batch_size,)``.
        target: Ground-truth values with the same shape as *pred*.
        l1_weight: Weight for the L1 component.  Defaults to 0.1.

    Returns:
        Scalar loss tensor.
    """
    mse_loss = F.mse_loss(pred, target)
    l1_loss = F.l1_loss(pred, target)
    return mse_loss + l1_weight * l1_loss


def ordinal_regression_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    biases: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ordinal regression loss with auxiliary bias ordering penalty.

    Uses binary cross-entropy with logits on each ordinal threshold, plus an
    auxiliary loss that encourages the bias terms to remain in descending
    order with a desired difference of 5.0.

    Args:
        logits: Model output of shape ``(batch_size, K)`` where
            ``K = num_classes - 1``.
        labels: Integer class labels of shape ``(batch_size,)`` with values in
            ``{0, 1, ..., num_classes - 1}``.
        biases: Ordinal bias parameters of shape ``(K,)``.
        sample_weights: Optional per-sample weights of shape
            ``(batch_size,)``.  When provided, the per-element BCE is
            multiplied by these weights before aggregation.

    Returns:
        Scalar loss tensor.
    """
    K = logits.size(1)
    indices = torch.arange(K, device=labels.device).unsqueeze(0)  # (1, K)
    targets = (labels.unsqueeze(1) > indices).float()  # (batch, K)

    bce_per_element = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )  # (batch, K)

    if sample_weights is not None:
        bce_per_element = bce_per_element * sample_weights.unsqueeze(1)

    out_loss = bce_per_element.sum(dim=1).mean()

    # Auxiliary loss to keep biases in descending order with controlled gaps.
    aux_loss = 0.0
    desired_diff = 5.0
    if len(biases) > 1:
        for i in range(len(biases) - 1):
            aux_loss = aux_loss + F.softplus(desired_diff - (biases[i] - biases[i + 1]))

    total_loss = out_loss + aux_loss
    return total_loss


def classification_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy classification loss.

    Args:
        pred: Logits of shape ``(batch_size, num_classes)``.
        target: Integer class labels of shape ``(batch_size,)``.

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(pred, target.long())


__all__ = [
    "regression_loss",
    "ordinal_regression_loss",
    "classification_loss",
]
