from __future__ import annotations

from querysense.evaluation.search_evaluator import evaluate_search_service
from querysense.evaluation.search_relevance import SearchRelevanceExample
from querysense.query_understanding.entities import ExtractedEntities
from querysense.retrieval.search_result import (
    ProductSearchResponse,
    ProductSearchResult,
)


class FakeSearchService:
    """Fake search service for search evaluator tests."""

    def search(self, query: str) -> ProductSearchResponse:
        if query == "headphones":
            results = [
                _build_result("p001"),
                _build_result("p002"),
            ]
        else:
            results = [
                _build_result("p999"),
                _build_result("p003"),
            ]

        return ProductSearchResponse(
            query=query,
            normalized_query=query,
            intent="product_search",
            entities=ExtractedEntities(),
            recommended_filters=[],
            results=results,
        )


def test_evaluate_search_service_returns_average_metrics() -> None:
    examples = [
        SearchRelevanceExample(
            query="headphones",
            relevant_product_ids={"p001"},
        ),
        SearchRelevanceExample(
            query="office laptop",
            relevant_product_ids={"p003"},
        ),
    ]

    result = evaluate_search_service(
        search_service=FakeSearchService(),  # type: ignore[arg-type]
        relevance_examples=examples,
        k=2,
    )

    assert result.summary.num_queries == 2
    assert result.summary.k == 2
    assert result.summary.precision_at_k == 0.5
    assert result.summary.recall_at_k == 1.0
    assert result.summary.mean_reciprocal_rank == 0.75
    assert 0.0 < result.summary.ndcg_at_k <= 1.0

    assert len(result.per_query_results) == 2
    assert set(result.per_query_results["query"]) == {"headphones", "office laptop"}


def test_evaluate_search_service_rejects_invalid_k() -> None:
    examples = [
        SearchRelevanceExample(
            query="headphones",
            relevant_product_ids={"p001"},
        )
    ]

    try:
        evaluate_search_service(
            search_service=FakeSearchService(),  # type: ignore[arg-type]
            relevance_examples=examples,
            k=0,
        )
    except ValueError as error:
        assert "k must be greater than 0" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_evaluate_search_service_handles_empty_examples() -> None:
    result = evaluate_search_service(
        search_service=FakeSearchService(),  # type: ignore[arg-type]
        relevance_examples=[],
        k=10,
    )

    assert result.summary.num_queries == 0
    assert result.summary.precision_at_k == 0.0
    assert result.summary.recall_at_k == 0.0
    assert result.summary.mean_reciprocal_rank == 0.0
    assert result.summary.ndcg_at_k == 0.0
    assert result.per_query_results.empty


def _build_result(product_id: str) -> ProductSearchResult:
    return ProductSearchResult(
        product_id=product_id,
        title=f"Product {product_id}",
        brand="Brand",
        category="Category",
        subcategory="Subcategory",
        color="Black",
        size="one-size",
        gender="unisex",
        condition="new",
        price=99.99,
        currency="EUR",
        score=1.0,
        bm25_score=0.0,
        semantic_score=0.0,
        match_reasons=[],
    )