"""Configuration layer: layered YAML config loading + shared constants.

Split out from the former ``src/core/`` (config and constants live here;
metrics have moved to ``src/metrics/``).

Layered loading order:
    configs/default.yaml  <  configs/bioparts/{biopart}.yaml  <  CLI args
"""

from .constants import SEQ_LENGTHS, NUCLEOTIDE_TO_INDEX, INDEX_TO_NUCLEOTIDE
from .config import (
    load_config,
    save_config,
    config_to_namespace,
    get_model_paths,
    get_predictor_configs,
    list_available_bioparts,
)

__all__ = [
    # constants
    "SEQ_LENGTHS",
    "NUCLEOTIDE_TO_INDEX",
    "INDEX_TO_NUCLEOTIDE",
    # config
    "load_config",
    "save_config",
    "config_to_namespace",
    "get_model_paths",
    "get_predictor_configs",
    "list_available_bioparts",
]
