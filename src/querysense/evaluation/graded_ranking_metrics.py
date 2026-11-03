
"""Graded ranking metrics for Learning-to-Rank evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class GradedRankingError(ValueError):
    """Raised when graded ranking inputs are invalid."""


def _validate_grades(
    relevance_grades: Sequence[int],
    k: int,
) -> list[int]:
    """Validate ranking grades and cutoff."""
    if k <= 0:
        raise GradedRankingError(
            "k must be greater than zero"
        )

    grades = list(relevance_grades)

    if any(
    isinstance(grade, bool | np.bool_)
    or not isinstance(grade, int | np.integer)
    or grade not in (0, 1, 2)
    for grade in grades
):
        raise GradedRankingError(
            "Relevance grades must be integers: 0, 1, or 2"
        )

    return grades


def discounted_cumulative_gain(
    relevance_grades: Sequence[int],
    k: int = 10,
) -> float:
    """Calculate DCG@K using exponential relevance gains.

    gain = 2 ** relevance_grade - 1

    Rank positions start at 1.
    """
    grades = _validate_grades(relevance_grades, k)

    dcg = 0.0

    for rank, grade in enumerate(grades[:k], start=1):
        gain = (2**grade) - 1
        discount = math.log2(rank + 1)

        dcg += gain / discount

    return float(dcg)


def graded_ndcg_at_k(
    relevance_grades: Sequence[int],
    k: int = 10,
) -> float:
    """Calculate graded NDCG@K for an already-ranked list.

    The input relevance grades must be ordered according
    to the ranking being evaluated.

    Returns zero if there are no positive relevance grades.
    """
    grades = _validate_grades(relevance_grades, k)

    if not grades:
        return 0.0

    dcg = discounted_cumulative_gain(
        grades,
        k=k,
    )

    ideal_grades = sorted(
        grades,
        reverse=True,
    )

    ideal_dcg = discounted_cumulative_gain(
        ideal_grades,
        k=k,
    )

    if ideal_dcg == 0.0:
        return 0.0

    return float(dcg / ideal_dcg)
