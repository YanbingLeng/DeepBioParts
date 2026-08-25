"""Iterative, continuous-coverage, non-repetitive library design."""

from .constraints import NrpSequenceConstraints
from .domain import DesignConfig, DesignedPart, DesignResult, GenerationFeedback
from .workflow import coverage_loss, coverage_radius, design_library, empirical_quantile

__all__ = [
    "DesignConfig",
    "DesignedPart",
    "DesignResult",
    "GenerationFeedback",
    "NrpSequenceConstraints",
    "coverage_loss",
    "coverage_radius",
    "design_library",
    "empirical_quantile",
]
