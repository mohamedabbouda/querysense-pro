from __future__ import annotations

from querysense.query_understanding.entities import ExtractedEntities
from querysense.query_understanding.filter_recommendation import FilterRecommendation
from querysense.retrieval.search_result import ProductSearchResponse, ProductSearchResult


def test_product_search_result_schema() -> None:
    result = ProductSearchResult(
        product_id="p001",
        title="Nike Air Zoom Pegasus",
        brand="nike",
        category="fashion",
        subcategory="shoes",
        color="black",
        size="44",
        gender="men",
        condition="new",
        price=89.99,
        currency="EUR",
        score=5.0,
        bm25_score=0.0,
        semantic_score=0.0,
        match_reasons=["brand", "subcategory"],
    )

    assert result.product_id == "p001"
    assert result.score == 5.0
    assert result.match_reasons == ["brand", "subcategory"]


def test_product_search_response_schema() -> None:
    result = ProductSearchResult(
        product_id="p001",
        title="Nike Air Zoom Pegasus",
        brand="nike",
        category="fashion",
        subcategory="shoes",
        color="black",
        size="44",
        gender="men",
        condition="new",
        price=89.99,
        currency="EUR",
        score=5.0,
        bm25_score=0.0,
        semantic_score=0.0,
        match_reasons=["brand", "subcategory"],
    )

    response = ProductSearchResponse(
        query="nike shoes",
        normalized_query="nike shoes",
        intent="product_search",
        entities=ExtractedEntities(brand="nike", subcategory="shoes"),
        recommended_filters=[
            FilterRecommendation(
                name="brand",
                value="nike",
                confidence=1.0,
                source="entity",
            )
        ],
        results=[result],
    )

    assert response.query == "nike shoes"
    assert response.intent == "product_search"
    assert response.entities.brand == "nike"
    assert response.recommended_filters[0].name == "brand"
    assert len(response.results) == 1