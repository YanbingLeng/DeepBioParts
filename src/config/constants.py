"""Shared constants for the DeepBioParts project.

This module centralizes all magic numbers, mappings, and default values
that were previously scattered across the codebase.
"""

from typing import Dict, FrozenSet, List, Tuple

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
# Default training parameters
# ---------------------------------------------------------------------------
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_KMER_SIZE: int = 3
DEFAULT_SIMILARITY_THRESHOLD: float = 0.8
DEFAULT_TRAIN_RATIO: float = 0.9

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
# Predictor model defaults
# ---------------------------------------------------------------------------
PREDICTOR_BATCH_SIZE: int = 64
PREDICTOR_LEARNING_RATE: float = 1e-4
PREDICTOR_WEIGHT_DECAY: float = 1e-4
PREDICTOR_PATIENCE: int = 100
PREDICTOR_NUM_EPOCHS: int = 200

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


# ---------------------------------------------------------------------------
# Predictor training defaults (training-script-specific)
# ---------------------------------------------------------------------------
# Default data paths, sequence lengths, early-stopping patience, and data augmentation
# strategies for each biopart's training data.
# Notes:
# - data_path is intentionally None: the training CSV must always be passed
#   explicitly via --data_path.
# - The hyperparameters and augmentation strategies here are not fully equivalent to
#   configs/bioparts/*.yaml; they are training-script-specific defaults, centralized
#   here for reuse.
BIOPART_DEFAULTS: Dict[str, dict] = {
    "promoter": {
        "data_path": None,
        "seq_len": 40,
        "patience": 100,
        "log_label": False,
        "use_cluster_split": True,
    },
    "rbs": {
        "data_path": None,
        "seq_len": 15,
        "patience": 100,
        "neighbor_interp": True,
        "interp_lambdas": [round(x / 20, 4) for x in range(1, 20)],
        "mutation_augment": False,
        "log_label": False,
        "use_cluster_split": True,
    },
    "terminator_regression": {
        "data_path": None,
        "seq_len": 50,
        "patience": 100,
        "neighbor_interp": True,
        "interp_lambdas": [round(x / 20, 4) for x in range(1, 20)],
        "mutation_augment": True,
        "mutation_n": 1,
        "mutation_copies": 2,
        "log_label": True,
        "use_cluster_split": True,
    },
    "terminator_classification": {
        "data_path": None,
        "seq_len": 50,
        "patience": 100,
    },
}

# Generic hyperparameter defaults (promoter / terminator)
NAIVE_DEFAULT_PARAMS: Dict[str, object] = {
    "learning_rate": 1e-4,
    "dropout_rate": 0.2,
    "batch_size": 64,
    "conv_width_motif": 5,
    "n_heads": 16,
    "conv_hidden": 128,
    "motif_conv_hidden": 256,
    "weight_decay": 1e-4,
    "optimizer": "adam",
}

# RBS-specific defaults (optimized for the augmentation-based training regime)
RBS_DEFAULT_PARAMS: Dict[str, object] = {
    "learning_rate": 1e-4,
    "dropout_rate": 0.15,
    "batch_size": 64,
    "conv_width_motif": 5,
    "n_heads": 8,
    "conv_hidden": 128,
    "motif_conv_hidden": 256,
    "weight_decay": 1e-4,
    "optimizer": "adam",
}


def get_biopart_defaults(biopart: str, task_type: str) -> dict:
    """Return the default configuration for a (biopart, task_type) combination."""
    key = biopart if biopart != "terminator" else f"terminator_{task_type}"
    if key not in BIOPART_DEFAULTS:
        key = "terminator_regression"
    return BIOPART_DEFAULTS[key]

