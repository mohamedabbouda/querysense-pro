from __future__ import annotations

from dataclasses import dataclass

from querysense.query_understanding.entities import ExtractedEntities
from querysense.query_understanding.filter_recommendation import FilterRecommendation


@dataclass(frozen=True)
class ProductSearchResult:
    """A product returned by the retrieval system."""

    product_id: str
    title: str
    brand: str
    category: str
    subcategory: str
    color: str
    size: str
    gender: str
    condition: str
    price: float
    currency: str
    score: float
    bm25_score: float
    semantic_score: float
    match_reasons: list[str]
    


@dataclass(frozen=True)
class ProductSearchResponse:
    """Search response containing query understanding and ranked product results."""

    query: str
    normalized_query: str
    intent: str
    entities: ExtractedEntities
    recommended_filters: list[FilterRecommendation]
    results: list[ProductSearchResult]