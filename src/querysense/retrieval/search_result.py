from __future__ import annotations

from dataclasses import dataclass


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
    match_reasons: list[str]


@dataclass(frozen=True)
class ProductSearchResponse:
    """Search response containing ranked product results."""

    query: str
    normalized_query: str
    intent: str
    results: list[ProductSearchResult]