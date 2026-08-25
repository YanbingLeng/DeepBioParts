"""Sequence encoding facade module.

Re-exports sequence encoding utilities from their original locations for
backward compatibility during migration.

Original sources:
    - ``data.sequence_encoding`` -- RNA tokenization encoding
    - ``utils.data`` -- one-hot encoding and reverse complement

Example:
    >>> from data.encoding import seq2onehot, onehot2seq, encode_sequences
    >>> oh = seq2onehot(["ACGT", "TGCA"])
    >>> seqs = onehot2seq(oh)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Re-export from utils.data: one-hot encoding helpers
# ---------------------------------------------------------------------------

try:
    from utils.data import seq2onehot, onehot2seq
except ImportError:
    seq2onehot = None  # type: ignore[assignment,misc]
    onehot2seq = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Re-export from utils.data: reverse complement
# ---------------------------------------------------------------------------
# NOTE: reverse_complement lives in utils.pc_util in the current codebase.

try:
    from utils.pc_util import reverse_complement  # type: ignore[no-redef]
except ImportError:
    try:
        from utils.data import reverse_complement  # type: ignore[no-redef]
    except ImportError:
        reverse_complement = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Re-export from data.sequence_encoding: tokenized encoding
# ---------------------------------------------------------------------------

try:
    from .sequence_encoding import (  # type: ignore[no-redef]
        encode_sequences,
        rna_clean_only,
        single_encoding,
        pairs_encoding,
        triples_encoding,
        ENCODING_FUNCTIONS,
    )
except ImportError:
    encode_sequences = None  # type: ignore[assignment,misc]
    rna_clean_only = None  # type: ignore[assignment,misc]
    single_encoding = None  # type: ignore[assignment,misc]
    pairs_encoding = None  # type: ignore[assignment,misc]
    triples_encoding = None  # type: ignore[assignment,misc]
    ENCODING_FUNCTIONS = None  # type: ignore[assignment,misc]

__all__ = [
    "seq2onehot",
    "onehot2seq",
    "reverse_complement",
    "encode_sequences",
    "rna_clean_only",
    "single_encoding",
    "pairs_encoding",
    "triples_encoding",
    "ENCODING_FUNCTIONS",
]
