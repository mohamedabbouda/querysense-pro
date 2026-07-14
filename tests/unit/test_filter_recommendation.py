from __future__ import annotations

from querysense.query_understanding.entities import ExtractedEntities
from querysense.query_understanding.filter_recommendation import (
    FilterRecommendation,
    FilterRecommendationResponse,
    recommend_filters_from_entities,
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


def test_recommend_filters_from_entities() -> None:
    entities = ExtractedEntities(
        brand="sony",
        subcategory="headphones",
        color="black",
        max_price=300.0,
    )

    filters = recommend_filters_from_entities(entities)

    assert [(filter_.name, filter_.value) for filter_ in filters] == [
        ("brand", "sony"),
        ("subcategory", "headphones"),
        ("color", "black"),
        ("max_price", 300.0),
    ]

    assert all(filter_.confidence == 1.0 for filter_ in filters)
    assert all(filter_.source == "entity" for filter_ in filters)


def test_recommend_filters_from_empty_entities_returns_empty_list() -> None:
    entities = ExtractedEntities()

    filters = recommend_filters_from_entities(entities)

    assert filters == []


def test_recommend_filters_includes_price_intent() -> None:
    entities = ExtractedEntities(
        subcategory="laptops",
        price_intent="cheap",
    )

    filters = recommend_filters_from_entities(entities)

    assert [(filter_.name, filter_.value) for filter_ in filters] == [
        ("subcategory", "laptops"),
        ("price_intent", "cheap"),
    ]