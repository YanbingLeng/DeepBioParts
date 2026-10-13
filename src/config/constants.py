"""Shared constants for the DeepBioParts project.

This module centralizes all magic numbers, mappings, and default values
that were previously scattered across the codebase.
"""

from typing import Dict, FrozenSet, Tuple

# ---------------------------------------------------------------------------
# Sequence lengths by biological part type
# ---------------------------------------------------------------------------
SEQ_LENGTHS: Dict[str, int] = {
    "promoter": 40,
    "rbs": 15,
    "terminator": 50,
}

# ---------------------------------------------------------------------------
# Nucleotide mappings
# ---------------------------------------------------------------------------
# Convention 1: ACGT order (alphabetical, used by sequence_encoding.py tokenization)
NUCLEOTIDE_TO_INDEX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
INDEX_TO_NUCLEOTIDE: Dict[int, str] = {0: "A", 1: "C", 2: "G", 3: "T"}

# Convention 2: TCGA order (used by seq2onehot() for both predictor and diffusion models)
ONEHOT_INDEX_TO_NUCLEOTIDE: Dict[int, str] = {0: "T", 1: "C", 2: "G", 3: "A"}
ONEHOT_NUCLEOTIDE_TO_INDEX: Dict[str, int] = {"T": 0, "C": 1, "G": 2, "A": 3}

# Valid DNA bases
VALID_BASES: FrozenSet[str] = frozenset({"A", "C", "G", "T"})

# Complement mapping
COMPLEMENT_MAP: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

# ---------------------------------------------------------------------------
# Diffusion model defaults
# ---------------------------------------------------------------------------
DIFFUSION_NUM_TIMESTEPS: int = 1000
DIFFUSION_BETA_START: float = 1e-4
DIFFUSION_BETA_END: float = 2e-2

# ---------------------------------------------------------------------------
# Diffusion model architecture defaults
# ---------------------------------------------------------------------------
DIFFUSION_SEQ_LEN: int = 40
DIFFUSION_TIME_EMB_DIM: int = 256
DIFFUSION_CHANNELS: int = 128
DIFFUSION_NUM_RESIDUAL_BLOCKS: int = 3
DIFFUSION_NUM_HEADS: int = 4

# ---------------------------------------------------------------------------
# GC content defaults
# ---------------------------------------------------------------------------
DEFAULT_GC_RANGE: Tuple[float, float] = (0.3, 0.7)
DEFAULT_POLY_STRENGTH: int = 5

# ---------------------------------------------------------------------------
# Publication figure settings
# ---------------------------------------------------------------------------
NATURE_DPI: int = 300
NATURE_FONT_FAMILY: str = "Arial"
NATURE_FONT_SIZE: int = 7
NATURE_LINE_WIDTH: float = 0.5
