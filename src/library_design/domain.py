"""Domain types for iterative regulatory-part library design."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GenerationFeedback:
    """Current repeat-constraint state supplied to the generator each round."""

    round_index: int
    kmer_size: int
    forbidden_kmers: frozenset[str]


class SequenceGenerator(Protocol):
    """Interface implemented by a trained sequence generator."""

    def generate(
        self,
        n_sequences: int,
        feedback: GenerationFeedback,
    ) -> Sequence[str]:
        """Generate candidate DNA sequences under the current feedback."""


class ActivityPredictor(Protocol):
    """Interface implemented by a trained sequence-to-activity predictor."""

    def predict(self, sequences: Sequence[str]) -> Sequence[float]:
        """Predict one activity value for each input sequence."""


@dataclass(frozen=True)
class DesignConfig:
    sequence_length: int
    target_library_size: int = 100
    lmax: int = 10
    generation_batch: int = 1_000
    max_rounds: int = 20
    max_swaps_per_round: int = 10
    min_coverage_gain: float = 0.0
    max_stagnant_rounds: int = 3

    def validate(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.target_library_size <= 0:
            raise ValueError("target_library_size must be positive")
        if self.lmax <= 4 or self.lmax >= self.sequence_length:
            raise ValueError(
                "lmax must satisfy 4 < lmax < sequence_length, as required by nrpcalc"
            )
        if self.generation_batch <= 0 or self.max_rounds <= 0:
            raise ValueError("generation_batch and max_rounds must be positive")
        if self.max_swaps_per_round < 0:
            raise ValueError("max_swaps_per_round cannot be negative")
        if self.min_coverage_gain < 0:
            raise ValueError("min_coverage_gain cannot be negative")
        if self.max_stagnant_rounds <= 0:
            raise ValueError("max_stagnant_rounds must be positive")


@dataclass(frozen=True)
class DesignedPart:
    sequence: str
    activity: float
    activity_quantile: float
    accepted_round: int


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    generated: int
    unique_valid: int
    finite_predictions: int
    swapped: int
    accepted: int
    library_size: int
    coverage_loss: float | None
    coverage_radius: float | None
    cumulative_predictions: int
    forbidden_kmers: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DesignResult:
    library: tuple[DesignedPart, ...]
    history: tuple[RoundRecord, ...]
    retired_sequences: frozenset[str]
    complete: bool
    coverage_loss: float | None
    coverage_radius: float | None
    cumulative_predictions: int
