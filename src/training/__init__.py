"""Training utilities: loss functions and early-stopping callbacks.

Predictor / diffusion-model training logic lives in ``naive_supervised_model``
and ``diffusion_model`` respectively (the former trainer_predictor /
trainer_diffusion were empty stubs pointing to them and have been removed -
the real implementation was never here).
"""

from .losses import regression_loss, ordinal_regression_loss, classification_loss
from .early_stopping import EarlyStopping_P, EarlyStopping_G

__all__ = [
    "regression_loss",
    "ordinal_regression_loss",
    "classification_loss",
    "EarlyStopping_P",
    "EarlyStopping_G",
]
