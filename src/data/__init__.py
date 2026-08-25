"""Data processing facade module.

Provides a unified import surface for sequence encoding, dataset classes,
data splitting, and I/O utilities.  Each sub-module re-exports symbols from
their original locations so that existing code continues to work during
migration.

Sub-modules:
    encoding: Sequence encoding (one-hot, tokenized, reverse complement).
    splitting: Cluster-based and random data splitting utilities.
"""

from .encoding import seq2onehot, onehot2seq, encode_sequences, reverse_complement
from .splitting import cluster_based_split, ClusterBasedKFold

__all__ = [
    # encoding
    "seq2onehot",
    "onehot2seq",
    "encode_sequences",
    "reverse_complement",
    # splitting
    "cluster_based_split",
    "ClusterBasedKFold",
]
