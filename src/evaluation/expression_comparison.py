#!/usr/bin/env python3
"""Pure computation functions for expression strength statistical comparison.

This module provides statistical metrics for comparing the expression strength
characteristics of two regulatory element libraries (e.g. a natural filtered
library vs. a generated library). All functions are pure computation with no
plotting dependencies; return values are structured data suitable for downstream
visualisation.

Metrics
-------
- Dynamic range and log-fold change
- Adjacent fitness gaps (sorted gaps between consecutive strengths)
- Distribution uniformity via Kolmogorov-Smirnov test
- eCDF area deviation from a perfect uniform distribution
- Full statistical report comparing two libraries
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_dynamic_range(
    strengths: np.ndarray,
) -> tuple[float, float]:
    """Compute the dynamic range of a set of expression strengths.

    The dynamic range is defined as the simple difference between the maximum
    and minimum strength values.  The log-fold change is
    ``log10(max / min)``, which quantifies how many orders of magnitude the
    library spans.

    Args:
        strengths: 1-D array of expression strength values.

    Returns:
        A tuple ``(dynamic_range, log_fold)`` where *dynamic_range* is
        ``max - min`` and *log_fold* is ``log10(max / min)``.  If the minimum
        value is zero or negative, *log_fold* is set to ``numpy.inf``.
    """
    s_min: float = float(np.min(strengths))
    s_max: float = float(np.max(strengths))
    dynamic_range: float = s_max - s_min
    log_fold: float = float(np.log10(s_max / s_min)) if s_min > 0 else np.inf
    return dynamic_range, log_fold


def compute_fitness_gaps(
    strengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute adjacent fitness gaps for a set of expression strengths.

    The strengths are sorted in ascending order and the pairwise differences
    between consecutive values are computed.  These gaps reveal "fault lines"
    in the activity landscape.

    Args:
        strengths: 1-D array of expression strength values.

    Returns:
        A tuple of five elements:
        - **sorted_s** (*np.ndarray*): Strength values sorted in ascending
          order.
        - **gaps** (*np.ndarray*): Adjacent gaps ``S[i+1] - S[i]`` for the
          sorted strengths (length ``n - 1``).
        - **max_gap** (*float*): Largest adjacent gap.
        - **mean_gap** (*float*): Mean of all adjacent gaps.
        - **median_gap** (*float*): Median of all adjacent gaps.
    """
    sorted_s: np.ndarray = np.sort(strengths)
    gaps: np.ndarray = np.diff(sorted_s)
    max_gap: float = float(np.max(gaps))
    mean_gap: float = float(np.mean(gaps))
    median_gap: float = float(np.median(gaps))
    return sorted_s, gaps, max_gap, mean_gap, median_gap


def compute_uniformity_ks(
    strengths: np.ndarray,
) -> tuple[float, float]:
    """Assess distribution uniformity via a Kolmogorov-Smirnov test.

    Strength values are min-max normalised to the interval [0, 1] and then
    tested against a ``Uniform(0, 1)`` reference distribution.  A smaller
    KS statistic indicates that the strengths are more uniformly spread
    across their range.

    Args:
        strengths: 1-D array of expression strength values.

    Returns:
        A tuple ``(ks_stat, p_value)`` where *ks_stat* is the KS test
        statistic (lower is more uniform) and *p_value* is the associated
        p-value.
    """
    normalized: np.ndarray = (strengths - np.min(strengths)) / (
        np.max(strengths) - np.min(strengths)
    )
    ks_stat: float
    p_value: float
    ks_stat, p_value = stats.kstest(normalized, "uniform")
    return ks_stat, p_value


def compute_ecdf_area_deviation(
    strengths: np.ndarray,
) -> float:
    """Compute the eCDF area deviation from a perfect uniform distribution.

    Strengths are min-max normalised to [0, 1] and sorted.  The empirical
    CDF is compared against the theoretical uniform CDF (a diagonal from
    (0, 0) to (1, 1)).  The returned value is the mean absolute deviation
    between the two curves — a smaller value indicates a more uniform
    distribution of strengths.

    Args:
        strengths: 1-D array of expression strength values.

    Returns:
        The mean absolute area deviation (*float*) between the empirical and
        ideal uniform CDFs.  Lower values indicate greater uniformity.
    """
    normalized: np.ndarray = np.sort(
        (strengths - np.min(strengths)) / (np.max(strengths) - np.min(strengths))
    )
    n: int = len(normalized)
    ideal: np.ndarray = np.arange(1, n + 1) / n
    area_dev: float = float(np.mean(np.abs(ideal - normalized)))
    return area_dev


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def full_statistical_report(
    nat_strengths: np.ndarray,
    gen_strengths: np.ndarray,
) -> dict:
    """Generate a full statistical comparison between two libraries.

    Computes all four metrics (dynamic range, fitness gaps, KS uniformity,
    and eCDF area deviation) for both the *natural* and *generated* strength
    arrays, as well as a Mann-Whitney U test comparing their gap
    distributions.

    Args:
        nat_strengths: 1-D array of expression strengths for the natural
            (filtered) library.
        gen_strengths: 1-D array of expression strengths for the generated
            library.

    Returns:
        A dictionary with the following keys:

        - ``nat_count`` / ``gen_count`` (*int*): Number of sequences.
        - ``nat_dynamic_range`` / ``gen_dynamic_range`` (*float*): Dynamic
          range (max - min).
        - ``nat_log_fold`` / ``gen_log_fold`` (*float*): Log-fold change.
        - ``nat_min`` / ``gen_min`` (*float*): Minimum strength.
        - ``nat_max`` / ``gen_max`` (*float*): Maximum strength.
        - ``nat_mean`` / ``gen_mean`` (*float*): Mean strength.
        - ``nat_max_gap`` / ``gen_max_gap`` (*float*): Largest adjacent gap.
        - ``nat_mean_gap`` / ``gen_mean_gap`` (*float*): Mean adjacent gap.
        - ``nat_median_gap`` / ``gen_median_gap`` (*float*): Median adjacent
          gap.
        - ``nat_gap_std`` / ``gen_gap_std`` (*float*): Standard deviation of
          gaps.
        - ``nat_gap_cv`` / ``gen_gap_cv`` (*float*): Coefficient of variation
          of gaps (std / mean).
        - ``nat_ks_stat`` / ``gen_ks_stat`` (*float*): KS test statistic.
        - ``nat_ks_p`` / ``gen_ks_p`` (*float*): KS test p-value.
        - ``nat_ecdf_area`` / ``gen_ecdf_area`` (*float*): eCDF area
          deviation.
        - ``mann_whitney_u`` (*float*): Mann-Whitney U statistic.
        - ``mann_whitney_p`` (*float*): Mann-Whitney p-value.
        - ``nat_sorted`` (*np.ndarray*): Sorted natural strengths.
        - ``gen_sorted`` (*np.ndarray*): Sorted generated strengths.
        - ``nat_gaps`` (*np.ndarray*): Natural adjacent gap array.
        - ``gen_gaps`` (*np.ndarray*): Generated adjacent gap array.
    """
    # Dynamic range
    nat_dr, nat_log = compute_dynamic_range(nat_strengths)
    gen_dr, gen_log = compute_dynamic_range(gen_strengths)

    # Fitness gaps
    nat_sorted, nat_gaps, nat_max_gap, nat_mean_gap, nat_med_gap = (
        compute_fitness_gaps(nat_strengths)
    )
    gen_sorted, gen_gaps, gen_max_gap, gen_mean_gap, gen_med_gap = (
        compute_fitness_gaps(gen_strengths)
    )

    # Uniformity (KS)
    nat_ks, nat_ks_p = compute_uniformity_ks(nat_strengths)
    gen_ks, gen_ks_p = compute_uniformity_ks(gen_strengths)

    # eCDF area deviation
    nat_area = compute_ecdf_area_deviation(nat_strengths)
    gen_area = compute_ecdf_area_deviation(gen_strengths)

    # Mann-Whitney U test: are natural gaps larger than generated gaps?
    mw_u, mw_p = stats.mannwhitneyu(nat_gaps, gen_gaps, alternative="greater")

    return {
        # Counts
        "nat_count": len(nat_strengths),
        "gen_count": len(gen_strengths),
        # Dynamic range
        "nat_dynamic_range": nat_dr,
        "gen_dynamic_range": gen_dr,
        "nat_log_fold": nat_log,
        "gen_log_fold": gen_log,
        # Min / Max / Mean
        "nat_min": float(np.min(nat_strengths)),
        "gen_min": float(np.min(gen_strengths)),
        "nat_max": float(np.max(nat_strengths)),
        "gen_max": float(np.max(gen_strengths)),
        "nat_mean": float(np.mean(nat_strengths)),
        "gen_mean": float(np.mean(gen_strengths)),
        # Gap statistics
        "nat_max_gap": nat_max_gap,
        "gen_max_gap": gen_max_gap,
        "nat_mean_gap": nat_mean_gap,
        "gen_mean_gap": gen_mean_gap,
        "nat_median_gap": nat_med_gap,
        "gen_median_gap": gen_med_gap,
        "nat_gap_std": float(np.std(nat_gaps)),
        "gen_gap_std": float(np.std(gen_gaps)),
        "nat_gap_cv": float(np.std(nat_gaps) / nat_mean_gap),
        "gen_gap_cv": float(np.std(gen_gaps) / gen_mean_gap),
        # KS uniformity
        "nat_ks_stat": nat_ks,
        "gen_ks_stat": gen_ks,
        "nat_ks_p": nat_ks_p,
        "gen_ks_p": gen_ks_p,
        # eCDF area deviation
        "nat_ecdf_area": nat_area,
        "gen_ecdf_area": gen_area,
        # Mann-Whitney U
        "mann_whitney_u": float(mw_u),
        "mann_whitney_p": float(mw_p),
        # Raw arrays (for downstream plotting)
        "nat_sorted": nat_sorted,
        "gen_sorted": gen_sorted,
        "nat_gaps": nat_gaps,
        "gen_gaps": gen_gaps,
    }
