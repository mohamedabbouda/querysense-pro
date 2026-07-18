from __future__ import annotations

import pytest

from querysense.evaluation.search_metrics import (
    evaluate_search_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k() -> None:
    precision = precision_at_k(
        retrieved_product_ids=["p001", "p002", "p003"],
        relevant_product_ids={"p001", "p004"},
        k=3,
    )

    assert precision == pytest.approx(1 / 3)


def test_recall_at_k() -> None:
    recall = recall_at_k(
        retrieved_product_ids=["p001", "p002", "p003"],
        relevant_product_ids={"p001", "p003", "p004"},
    )

    assert recall == pytest.approx(2 / 3)


def test_reciprocal_rank_returns_inverse_rank_of_first_relevant_result() -> None:
    score = reciprocal_rank(
        retrieved_product_ids=["p010", "p002", "p003"],
        relevant_product_ids={"p003"},
    )

    assert score == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_no_relevant_result_found() -> None:
    score = reciprocal_rank(
        retrieved_product_ids=["p010", "p002", "p003"],
        relevant_product_ids={"p999"},
    )

    assert score == 0.0


def test_ndcg_at_k_is_one_for_ideal_ranking() -> None:
    score = ndcg_at_k(
        retrieved_product_ids=["p001", "p002", "p003"],
        relevant_product_ids={"p001", "p002"},
        k=3,
    )

    assert score == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_lower_ranked_relevant_results() -> None:
    score = ndcg_at_k(
        retrieved_product_ids=["p999", "p001", "p002"],
        relevant_product_ids={"p001", "p002"},
        k=3,
    )

    assert 0.0 < score < 1.0


def test_evaluate_search_ranking_returns_all_metrics() -> None:
    result = evaluate_search_ranking(
        retrieved_product_ids=["p999", "p001", "p002"],
        relevant_product_ids={"p001", "p002"},
        k=3,
    )

    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.reciprocal_rank == pytest.approx(1 / 2)
    assert 0.0 < result.ndcg_at_k < 1.0


def test_precision_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be greater than 0"):
        precision_at_k(
            retrieved_product_ids=["p001"],
            relevant_product_ids={"p001"},
            k=0,
        )


def test_ndcg_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be greater than 0"):
        ndcg_at_k(
            retrieved_product_ids=["p001"],
            relevant_product_ids={"p001"},
            k=0,
        )