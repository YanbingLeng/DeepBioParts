"""Adapter from the design workflow to the bundled NRP Calculator."""

from __future__ import annotations

import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence


class SequenceConstraintStore(Protocol):
    """Incremental within- and between-sequence repeat boundary."""

    @property
    def lmax(self) -> int: ...

    @property
    def forbidden_kmers(self) -> frozenset[str]: ...

    def add(self, sequence: str) -> None: ...

    def remove(self, sequence: str) -> None: ...

    def conflicts(self, sequence: str) -> frozenset[str]: ...

    def validate_library(self, sequences: Sequence[str]) -> bool: ...


@lru_cache(maxsize=1)
def _load_bundled_nrpcalc():
    """Load nrpcalc specifically from ``src/nrpcalc``."""

    source_root = Path(__file__).resolve().parents[1] / "nrpcalc"
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    try:
        import nrpcalc
        from nrpcalc.base.utils import stream_min_kmers
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import the bundled src/nrpcalc implementation. Activate the "
            "project environment or run `python -m pip install -e src/nrpcalc`. "
            f"Original import error: {exc}"
        ) from exc

    module_path = Path(nrpcalc.__file__).resolve()
    if source_root.resolve() not in module_path.parents:
        raise RuntimeError(
            f"Imported nrpcalc from {module_path}, expected the bundled copy under "
            f"{source_root.resolve()}"
        )
    return nrpcalc, stream_min_kmers


def nrp_kmers(sequence: str, k: int) -> frozenset[str]:
    """Return nrpcalc canonical k-mers, including reverse-complement equivalence."""

    if k <= 0:
        raise ValueError("k must be positive")
    _, stream_min_kmers = _load_bundled_nrpcalc()
    return frozenset(stream_min_kmers(sequence.replace("U", "T"), k))


_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def has_internal_repeat(sequence: str, k: int) -> bool:
    """Return whether a canonical k-mer occurs twice within one sequence."""

    if k <= 0:
        raise ValueError("k must be positive")

    normalized = sequence.upper().replace("U", "T")
    seen: set[str] = set()
    for start in range(len(normalized) - k + 1):
        kmer = normalized[start : start + k]
        reverse_complement = kmer.translate(_DNA_COMPLEMENT)[::-1]
        canonical = min(kmer, reverse_complement)
        if canonical in seen:
            return True
        seen.add(canonical)
    return False


class NrpSequenceConstraints:
    """On-disk NRP background plus owner attribution for iterative eviction.

    ``nrpcalc.background`` is the source of truth for conflict decisions. The
    in-memory owner table only identifies which accepted sequence owns a
    conflicting canonical k-mer, information needed by the eviction policy.
    """

    def __init__(self, lmax: int) -> None:
        if lmax <= 4:
            raise ValueError("nrpcalc requires lmax > 4")

        self._lmax = lmax
        self._k = lmax + 1
        self._api, self._stream_min_kmers = _load_bundled_nrpcalc()
        temp_root = Path(__file__).resolve().parents[2] / ".tmp" / "nrpcalc"
        temp_root.mkdir(parents=True, exist_ok=True)
        self._storage_path = Path(
            tempfile.mkdtemp(prefix="background_", dir=temp_root)
        )
        self._background = self._api.background(
            path=str(self._storage_path),
            Lmax=lmax,
            verbose=False,
        )
        self._sequences: set[str] = set()
        self._kmer_owner: dict[str, str] = {}
        self._closed = False

    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def forbidden_kmers(self) -> frozenset[str]:
        self._require_open()
        return frozenset(self._background)

    def _kmers(self, sequence: str) -> frozenset[str]:
        return frozenset(self._stream_min_kmers(sequence.replace("U", "T"), self._k))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("NRP constraint store is closed")

    def add(self, sequence: str) -> None:
        self._require_open()
        if sequence in self._sequences:
            raise ValueError(f"sequence already present: {sequence}")
        if has_internal_repeat(sequence, self._k):
            raise ValueError(
                f"sequence contains an internal repeat longer than Lmax={self._lmax}"
            )
        blockers = self.conflicts(sequence)
        if blockers:
            raise ValueError(
                f"sequence violates Lmax={self._lmax}; blocked by {sorted(blockers)}"
            )

        self._background.add(sequence)
        for kmer in self._kmers(sequence):
            self._kmer_owner[kmer] = sequence
        self._sequences.add(sequence)

    def remove(self, sequence: str) -> None:
        self._require_open()
        if sequence not in self._sequences:
            raise KeyError(sequence)

        self._background.remove(sequence)
        for kmer in self._kmers(sequence):
            if self._kmer_owner.get(kmer) == sequence:
                del self._kmer_owner[kmer]
        self._sequences.remove(sequence)

    def conflicts(self, sequence: str) -> frozenset[str]:
        self._require_open()
        if sequence not in self._background:
            return frozenset()

        blockers = frozenset(
            self._kmer_owner[kmer]
            for kmer in self._kmers(sequence)
            if kmer in self._kmer_owner
        )
        if not blockers:
            raise RuntimeError("nrpcalc background and owner attribution are out of sync")
        return blockers

    def validate_library(self, sequences: Sequence[str]) -> bool:
        """Use NRP Finder Mode as an independent final-library check."""

        self._require_open()
        sequence_list = list(sequences)
        if any(has_internal_repeat(sequence, self._k) for sequence in sequence_list):
            return False
        if len(sequence_list) < 2:
            return True
        non_repetitive = self._api.finder(
            seq_list=sequence_list,
            Lmax=self._lmax,
            internal_repeats=False,
            background=None,
            verbose=False,
        )
        return len(non_repetitive) == len(sequence_list)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._background.drop()
        finally:
            self._closed = True

    def __enter__(self) -> "NrpSequenceConstraints":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
