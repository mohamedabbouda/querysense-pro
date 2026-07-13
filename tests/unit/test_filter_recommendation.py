from __future__ import annotations

from querysense.query_understanding.filter_recommendation import (
    FilterRecommendation,
    FilterRecommendationResponse,
)


def test_filter_recommendation_schema() -> None:
    recommendation = FilterRecommendation(
        name="brand",
        value="sony",
        confidence=1.0,
        source="entity",
    )

    assert recommendation.name == "brand"
    assert recommendation.value == "sony"
    assert recommendation.confidence == 1.0
    assert recommendation.source == "entity"


def test_filter_recommendation_response_schema() -> None:
    recommendation = FilterRecommendation(
        name="brand",
        value="sony",
        confidence=1.0,
        source="entity",
    )

    response = FilterRecommendationResponse(
        query="sony headphones",
        normalized_query="sony headphones",
        filters=[recommendation],
    )

    assert response.query == "sony headphones"
    assert response.normalized_query == "sony headphones"
    assert len(response.filters) == 1