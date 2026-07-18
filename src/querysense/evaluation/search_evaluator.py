from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from querysense.evaluation.search_metrics import (
    evaluate_search_ranking,
)
from querysense.evaluation.search_relevance import SearchRelevanceExample
from querysense.retrieval.search_service import ProductSearchService


@dataclass(frozen=True)
class SearchEvaluationSummary:
    """Average search metrics over a relevance benchmark."""

    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    num_queries: int
    k: int


@dataclass(frozen=True)
class SearchEvaluationResult:
    """Full search evaluation result."""

    summary: SearchEvaluationSummary
    per_query_results: pd.DataFrame


def evaluate_search_service(
    search_service: ProductSearchService,
    relevance_examples: list[SearchRelevanceExample],
    k: int = 10,
) -> SearchEvaluationResult:
    """Evaluate a product search service over relevance examples."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    per_query_rows = []

    for example in relevance_examples:
        response = search_service.search(example.query)
        retrieved_product_ids = [
            result.product_id
            for result in response.results
        ]

        metrics = evaluate_search_ranking(
            retrieved_product_ids=retrieved_product_ids,
            relevant_product_ids=example.relevant_product_ids,
            k=k,
        )

        per_query_rows.append(
            {
                "query": example.query,
                "relevant_product_ids": sorted(example.relevant_product_ids),
                "retrieved_product_ids": retrieved_product_ids[:k],
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "reciprocal_rank": metrics.reciprocal_rank,
                "ndcg_at_k": metrics.ndcg_at_k,
            }
        )

    per_query_results = pd.DataFrame(per_query_rows)

    summary = _summarize_results(
        per_query_results=per_query_results,
        k=k,
    )

    return SearchEvaluationResult(
        summary=summary,
        per_query_results=per_query_results,
    )


def _summarize_results(
    per_query_results: pd.DataFrame,
    k: int,
) -> SearchEvaluationSummary:
    if per_query_results.empty:
        return SearchEvaluationSummary(
            precision_at_k=0.0,
            recall_at_k=0.0,
            mean_reciprocal_rank=0.0,
            ndcg_at_k=0.0,
            num_queries=0,
            k=k,
        )

    return SearchEvaluationSummary(
        precision_at_k=float(per_query_results["precision_at_k"].mean()),
        recall_at_k=float(per_query_results["recall_at_k"].mean()),
        mean_reciprocal_rank=float(per_query_results["reciprocal_rank"].mean()),
        ndcg_at_k=float(per_query_results["ndcg_at_k"].mean()),
        num_queries=len(per_query_results),
        k=k,
    )