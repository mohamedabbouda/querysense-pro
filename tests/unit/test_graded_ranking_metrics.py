
from __future__ import annotations

import pytest

from querysense.evaluation.graded_ranking_metrics import (
    GradedRankingError,
    discounted_cumulative_gain,
    graded_ndcg_at_k,
)


def test_perfect_ranking_has_ndcg_one() -> None:
    grades = [2, 1, 0]

    assert graded_ndcg_at_k(
        grades,
        k=3,
    ) == pytest.approx(1.0)


def test_imperfect_ranking_has_lower_ndcg() -> None:
    perfect = graded_ndcg_at_k(
        [2, 1, 0],
        k=3,
    )

    imperfect = graded_ndcg_at_k(
        [1, 2, 0],
        k=3,
    )

    assert imperfect < perfect
    assert imperfect == pytest.approx(0.7967, abs=0.001)


def test_all_irrelevant_returns_zero() -> None:
    assert graded_ndcg_at_k(
        [0, 0, 0],
        k=3,
    ) == 0.0


def test_empty_ranking_returns_zero() -> None:
    assert graded_ndcg_at_k([], k=10) == 0.0


def test_ndcg_respects_cutoff() -> None:
    grades = [2, 1, 0, 2]

    result = graded_ndcg_at_k(
        grades,
        k=2,
    )

    # The first two grades determine DCG.
    actual_dcg = discounted_cumulative_gain(
        [2, 1],
        k=2,
    )

    # But IDCG considers the entire candidate pool.
    ideal_dcg = discounted_cumulative_gain(
        [2, 2, 1, 0],
        k=2,
    )

    assert result == pytest.approx(
        actual_dcg / ideal_dcg
    )


def test_rejects_invalid_grades() -> None:
    with pytest.raises(
        GradedRankingError,
        match="Relevance grades",
    ):
        graded_ndcg_at_k([0, 1, 3], k=3)


def test_rejects_invalid_k() -> None:
    with pytest.raises(
        GradedRankingError,
        match="k must be greater",
    ):
        graded_ndcg_at_k([2, 1, 0], k=0)


def test_graded_dcg_rewards_exact_more_than_partial() -> None:
    exact_first = discounted_cumulative_gain(
        [2, 1],
        k=2,
    )

    partial_first = discounted_cumulative_gain(
        [1, 2],
        k=2,
    )

    assert exact_first > partial_first
