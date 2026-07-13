from __future__ import annotations

from dataclasses import dataclass


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