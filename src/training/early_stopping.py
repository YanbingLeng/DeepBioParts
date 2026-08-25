"""Early stopping utilities facade module.

Re-exports early stopping callback classes from ``utils.utils`` for backward
compatibility during migration.

Classes:
    EarlyStopping_P: Early stopping based on a single validation metric
        (supports both ``'max'`` and ``'min'`` ordering).
    EarlyStopping_G: Extended early stopping that monitors JS divergence,
        accuracy, and reconstruction loss simultaneously.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export from utils.utils
# ---------------------------------------------------------------------------

try:
    from utils.utils import EarlyStopping_P, EarlyStopping_G  # type: ignore[no-redef]
except ImportError:
    EarlyStopping_P = None  # type: ignore[assignment,misc]
    EarlyStopping_G = None  # type: ignore[assignment,misc]

__all__ = [
    "EarlyStopping_P",
    "EarlyStopping_G",
]
