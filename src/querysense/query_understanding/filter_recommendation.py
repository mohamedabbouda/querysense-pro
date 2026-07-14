from __future__ import annotations

from dataclasses import dataclass

from querysense.query_understanding.entities import ExtractedEntities


@dataclass(frozen=True)
class FilterRecommendation:
    """A recommended product search filter."""

    name: str
    value: str | float
    confidence: float
    source: str


@dataclass(frozen=True)
class FilterRecommendationResponse:
    """Filter recommendation response for a query."""

    query: str
    normalized_query: str
    filters: list[FilterRecommendation]


def recommend_filters_from_entities(
    entities: ExtractedEntities,
) -> list[FilterRecommendation]:
    """Convert extracted query entities into recommended product filters."""
    filters: list[FilterRecommendation] = []

    _add_filter(filters, "brand", entities.brand)
    _add_filter(filters, "category", entities.category)
    _add_filter(filters, "subcategory", entities.subcategory)
    _add_filter(filters, "color", entities.color)
    _add_filter(filters, "size", entities.size)
    _add_filter(filters, "gender", entities.gender)
    _add_filter(filters, "condition", entities.condition)
    _add_filter(filters, "min_price", entities.min_price)
    _add_filter(filters, "max_price", entities.max_price)
    _add_filter(filters, "price_intent", entities.price_intent)

    return filters


def _add_filter(
    filters: list[FilterRecommendation],
    name: str,
    value: str | float | None,
) -> None:
    if value is None:
        return

    filters.append(
        FilterRecommendation(
            name=name,
            value=value,
            confidence=1.0,
            source="entity",
        )
    )