from __future__ import annotations

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
        match_reasons=["brand", "subcategory"],
    )

    response = ProductSearchResponse(
        query="nike shoes",
        normalized_query="nike shoes",
        intent="product_search",
        results=[result],
    )

    assert response.query == "nike shoes"
    assert response.intent == "product_search"
    assert len(response.results) == 1