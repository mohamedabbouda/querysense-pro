from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchMetricResult:
    """Ranking metrics for one search query."""

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def evaluate_search_ranking(
    retrieved_product_ids: list[str],
    relevant_product_ids: set[str],
    k: int = 10,
) -> SearchMetricResult:
    """Evaluate ranking quality for one query."""
    top_k_product_ids = retrieved_product_ids[:k]

    return SearchMetricResult(
        precision_at_k=precision_at_k(
            retrieved_product_ids=top_k_product_ids,
            relevant_product_ids=relevant_product_ids,
            k=k,
        ),
        recall_at_k=recall_at_k(
            retrieved_product_ids=top_k_product_ids,
            relevant_product_ids=relevant_product_ids,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_product_ids=retrieved_product_ids,
            relevant_product_ids=relevant_product_ids,
        ),
        ndcg_at_k=ndcg_at_k(
            retrieved_product_ids=top_k_product_ids,
            relevant_product_ids=relevant_product_ids,
            k=k,
        ),
    )


def precision_at_k(
    retrieved_product_ids: list[str],
    relevant_product_ids: set[str],
    k: int,
) -> float:
    """Compute Precision@K."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not retrieved_product_ids:
        return 0.0

    top_k_product_ids = retrieved_product_ids[:k]
    relevant_retrieved_count = _count_relevant_retrieved(
        retrieved_product_ids=top_k_product_ids,
        relevant_product_ids=relevant_product_ids,
    )

    return relevant_retrieved_count / k


def recall_at_k(
    retrieved_product_ids: list[str],
    relevant_product_ids: set[str],
) -> float:
    """Compute Recall@K."""
    if not relevant_product_ids:
        return 0.0

    relevant_retrieved_count = _count_relevant_retrieved(
        retrieved_product_ids=retrieved_product_ids,
        relevant_product_ids=relevant_product_ids,
    )

    return relevant_retrieved_count / len(relevant_product_ids)


def reciprocal_rank(
    retrieved_product_ids: list[str],
    relevant_product_ids: set[str],
) -> float:
    """Compute reciprocal rank for the first relevant result."""
    for index, product_id in enumerate(retrieved_product_ids, start=1):
        if product_id in relevant_product_ids:
            return 1.0 / index

    return 0.0


def ndcg_at_k(
    retrieved_product_ids: list[str],
    relevant_product_ids: set[str],
    k: int,
) -> float:
    """Compute binary NDCG@K."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_product_ids:
        return 0.0

    top_k_product_ids = retrieved_product_ids[:k]

    dcg = _discounted_cumulative_gain(
        retrieved_product_ids=top_k_product_ids,
        relevant_product_ids=relevant_product_ids,
    )

    ideal_relevant_count = min(len(relevant_product_ids), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def _discounted_cumulative_gain(
    retrieved_product_ids: list[str],
    relevant_product_ids: set[str],
) -> float:
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, product_id in enumerate(retrieved_product_ids, start=1)
        if product_id in relevant_product_ids
    )


def _count_relevant_retrieved(
    retrieved_product_ids: list[str],
    relevant_product_ids: set[str],
) -> int:
    return sum(
        1
        for product_id in retrieved_product_ids
        if product_id in relevant_product_ids
    )