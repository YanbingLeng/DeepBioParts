"""Iterative DDPM-predictor workflow for continuous activity coverage."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import replace
from typing import Sequence

from .constraints import SequenceConstraintStore, has_internal_repeat
from .domain import (
    ActivityPredictor,
    DesignConfig,
    DesignedPart,
    DesignResult,
    GenerationFeedback,
    RoundRecord,
    SequenceGenerator,
)


DNA_ALPHABET = frozenset("ACGT")


class _LibraryState:
    """Keep part records synchronized with the repeat-constraint store."""

    def __init__(self, constraints: SequenceConstraintStore) -> None:
        self.constraints = constraints
        self._parts: dict[str, DesignedPart] = {}

    @property
    def parts(self) -> tuple[DesignedPart, ...]:
        return tuple(self._parts.values())

    def __contains__(self, sequence: str) -> bool:
        return sequence in self._parts

    def get(self, sequence: str) -> DesignedPart:
        return self._parts[sequence]

    def conflicts(self, sequence: str) -> frozenset[str]:
        return self.constraints.conflicts(sequence)

    def add(self, part: DesignedPart) -> None:
        self.constraints.add(part.sequence)
        self._parts[part.sequence] = part

    def remove(self, sequence: str) -> DesignedPart:
        self.constraints.remove(sequence)
        return self._parts.pop(sequence)

    def refresh_quantiles(self, sorted_reference: Sequence[float]) -> None:
        self._parts = {
            part.sequence: replace(
                part,
                activity_quantile=empirical_quantile(
                    part.activity,
                    sorted_reference,
                ),
            )
            for part in self.parts
        }


def empirical_quantile(activity: float, sorted_reference: Sequence[float]) -> float:
    """Map an activity to its mid-rank empirical-CDF coordinate in ``(0, 1)``."""

    if not sorted_reference:
        raise ValueError("empirical quantiles require at least one reference value")
    left = bisect_left(sorted_reference, activity)
    right = bisect_right(sorted_reference, activity)
    return (left + right) / (2.0 * len(sorted_reference))


def coverage_loss(quantiles: Sequence[float]) -> float | None:
    """Return the exact integrated squared distance to the nearest quantile."""

    if not quantiles:
        return None
    ordered = sorted(quantiles)
    loss = ordered[0] ** 3 / 3.0
    loss += (1.0 - ordered[-1]) ** 3 / 3.0
    loss += sum(
        (right - left) ** 3 / 12.0
        for left, right in zip(ordered, ordered[1:])
    )
    return loss


def coverage_radius(quantiles: Sequence[float]) -> float | None:
    """Return the largest distance from ``[0, 1]`` to the nearest library member."""

    if not quantiles:
        return None
    ordered = sorted(quantiles)
    interior = (
        max((right - left) / 2.0 for left, right in zip(ordered, ordered[1:]))
        if len(ordered) > 1
        else 0.0
    )
    return max(ordered[0], 1.0 - ordered[-1], interior)


def marginal_coverage_gain(
    library_quantiles: Sequence[float],
    candidate_quantile: float,
) -> float:
    """Return the exact reduction in coverage loss from adding one candidate."""

    ordered = sorted(library_quantiles)
    if not ordered:
        singleton_loss = coverage_loss((candidate_quantile,))
        if singleton_loss is None:
            raise RuntimeError("singleton coverage loss is undefined")
        return 1.0 / 3.0 - singleton_loss

    position = bisect_left(ordered, candidate_quantile)
    if position == 0:
        right = ordered[0]
        before = right**3 / 3.0
        after = candidate_quantile**3 / 3.0
        after += (right - candidate_quantile) ** 3 / 12.0
        return before - after
    if position == len(ordered):
        left = ordered[-1]
        before = (1.0 - left) ** 3 / 3.0
        after = (candidate_quantile - left) ** 3 / 12.0
        after += (1.0 - candidate_quantile) ** 3 / 3.0
        return before - after

    left = ordered[position - 1]
    right = ordered[position]
    before = (right - left) ** 3 / 12.0
    after = (candidate_quantile - left) ** 3 / 12.0
    after += (right - candidate_quantile) ** 3 / 12.0
    return before - after


def _normalize_candidates(
    sequences: Sequence[str],
    sequence_length: int,
    excluded: set[str],
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in sequences:
        sequence = str(raw).strip().upper().replace("U", "T")
        if (
            len(sequence) != sequence_length
            or not set(sequence).issubset(DNA_ALPHABET)
            or sequence in excluded
            or sequence in seen
        ):
            continue
        seen.add(sequence)
        normalized.append(sequence)
    return normalized


def _accept_coverage_improving_candidates(
    library: _LibraryState,
    candidates: Sequence[DesignedPart],
    config: DesignConfig,
) -> tuple[list[DesignedPart], list[DesignedPart]]:
    """Greedily add conflict-free candidates with the largest coverage gain."""

    remaining = {candidate.sequence: candidate for candidate in candidates}
    accepted: list[DesignedPart] = []

    while remaining and len(library.parts) < config.target_library_size:
        library_quantiles = [part.activity_quantile for part in library.parts]
        best: DesignedPart | None = None
        best_key: tuple[float, str] | None = None

        for candidate in remaining.values():
            if library.conflicts(candidate.sequence):
                continue
            gain = marginal_coverage_gain(
                library_quantiles,
                candidate.activity_quantile,
            )
            key = (gain, candidate.sequence)
            if best_key is None or key > best_key:
                best = candidate
                best_key = key

        if best is None or best_key is None:
            break
        if best_key[0] <= config.min_coverage_gain:
            break

        library.add(best)
        accepted.append(best)
        del remaining[best.sequence]

    return accepted, list(remaining.values())


def _swap_coverage_improving_candidates(
    library: _LibraryState,
    candidates: Sequence[DesignedPart],
    config: DesignConfig,
    retired_sequences: set[str],
) -> list[DesignedPart]:
    """Apply positive-gain one-for-one replacements.

    A candidate with one conflict must replace that blocker. Once the library is
    full, a conflict-free candidate may replace any retained member.
    """

    remaining = {candidate.sequence: candidate for candidate in candidates}
    swapped_in: list[DesignedPart] = []

    for _ in range(config.max_swaps_per_round):
        current_parts = library.parts
        current_loss = coverage_loss(
            [part.activity_quantile for part in current_parts]
        )
        if current_loss is None:
            break

        best_candidate: DesignedPart | None = None
        best_blocker: str | None = None
        best_key: tuple[float, str, str] | None = None

        for candidate in remaining.values():
            blockers = library.conflicts(candidate.sequence)
            if len(blockers) > 1:
                continue
            if blockers:
                removable = blockers
            elif len(current_parts) >= config.target_library_size:
                removable = frozenset(part.sequence for part in current_parts)
            else:
                continue

            for blocker in removable:
                replacement_quantiles = [
                    part.activity_quantile
                    for part in current_parts
                    if part.sequence != blocker
                ]
                replacement_quantiles.append(candidate.activity_quantile)
                replacement_loss = coverage_loss(replacement_quantiles)
                if replacement_loss is None:
                    continue
                gain = current_loss - replacement_loss
                key = (gain, candidate.sequence, blocker)
                if best_key is None or key > best_key:
                    best_candidate = candidate
                    best_blocker = blocker
                    best_key = key

        if (
            best_candidate is None
            or best_blocker is None
            or best_key is None
            or best_key[0] <= config.min_coverage_gain
        ):
            break

        removed = library.remove(best_blocker)
        retired_sequences.add(removed.sequence)
        library.add(best_candidate)
        swapped_in.append(best_candidate)
        del remaining[best_candidate.sequence]

    return swapped_in


def design_library(
    generator: SequenceGenerator,
    predictor: ActivityPredictor,
    config: DesignConfig,
    constraints: SequenceConstraintStore,
) -> DesignResult:
    """Select continuous coverage under within- and between-sequence constraints."""

    config.validate()
    if constraints.lmax != config.lmax:
        raise ValueError(
            f"constraint Lmax ({constraints.lmax}) does not match config ({config.lmax})"
        )

    library = _LibraryState(constraints)
    retired_sequences: set[str] = set()
    activity_reference: list[float] = []
    history: list[RoundRecord] = []
    stagnant_rounds = 0

    for round_index in range(1, config.max_rounds + 1):
        feedback = GenerationFeedback(
            round_index=round_index,
            kmer_size=config.lmax + 1,
            forbidden_kmers=constraints.forbidden_kmers,
        )
        generated = list(generator.generate(config.generation_batch, feedback))
        excluded = set(retired_sequences)
        excluded.update(part.sequence for part in library.parts)
        unique_valid = _normalize_candidates(
            generated,
            config.sequence_length,
            excluded,
        )
        unique_valid = [
            sequence
            for sequence in unique_valid
            if not has_internal_repeat(sequence, config.lmax + 1)
        ]

        activities = list(predictor.predict(unique_valid)) if unique_valid else []
        if len(activities) != len(unique_valid):
            raise ValueError(
                "predictor returned a different number of activities than sequences"
            )

        eligible = [
            (sequence, float(activity))
            for sequence, activity in zip(unique_valid, activities)
            if math.isfinite(float(activity))
        ]
        activity_reference.extend(activity for _, activity in eligible)
        sorted_reference = sorted(activity_reference)

        if sorted_reference:
            library.refresh_quantiles(sorted_reference)
        candidates = [
            DesignedPart(
                sequence=sequence,
                activity=activity,
                activity_quantile=empirical_quantile(activity, sorted_reference),
                accepted_round=round_index,
            )
            for sequence, activity in eligible
        ]

        accepted, remaining = _accept_coverage_improving_candidates(
            library,
            candidates,
            config,
        )
        swapped_in = _swap_coverage_improving_candidates(
            library,
            remaining,
            config,
            retired_sequences,
        )

        changed = len(accepted) + len(swapped_in)
        stagnant_rounds = stagnant_rounds + 1 if changed == 0 else 0
        quantiles = [part.activity_quantile for part in library.parts]
        round_loss = coverage_loss(quantiles)
        round_radius = coverage_radius(quantiles)
        history.append(
            RoundRecord(
                round_index=round_index,
                generated=len(generated),
                unique_valid=len(unique_valid),
                finite_predictions=len(candidates),
                swapped=len(swapped_in),
                accepted=len(accepted),
                library_size=len(library.parts),
                coverage_loss=round_loss,
                coverage_radius=round_radius,
                cumulative_predictions=len(activity_reference),
                forbidden_kmers=len(constraints.forbidden_kmers),
            )
        )

        if stagnant_rounds >= config.max_stagnant_rounds:
            break

    designed_parts = tuple(sorted(library.parts, key=lambda part: part.activity))
    if not constraints.validate_library(
        [part.sequence for part in designed_parts]
    ):
        raise RuntimeError(
            "final library violates a within- or between-sequence Lmax constraint"
        )

    final_quantiles = [part.activity_quantile for part in designed_parts]
    return DesignResult(
        library=designed_parts,
        history=tuple(history),
        retired_sequences=frozenset(retired_sequences),
        complete=len(designed_parts) >= config.target_library_size,
        coverage_loss=coverage_loss(final_quantiles),
        coverage_radius=coverage_radius(final_quantiles),
        cumulative_predictions=len(activity_reference),
    )
