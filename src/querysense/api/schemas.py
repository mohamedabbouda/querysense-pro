from __future__ import annotations

from pydantic import BaseModel, Field


class IntentPredictionRequest(BaseModel):
    """Request body for intent prediction."""

    query: str = Field(..., min_length=1)


class IntentPredictionResponse(BaseModel):
    """Response body for intent prediction."""

    query: str
    normalized_query: str
    intent: str
    source: str
    model_intent: str | None


class ProductSearchRequest(BaseModel):
    """Request body for product search."""

    query: str = Field(..., min_length=1)


class ProductSearchResultResponse(BaseModel):
    """Single product search result response."""

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


class ProductSearchResponseModel(BaseModel):
    """Product search API response."""

    query: str
    normalized_query: str
    intent: str
    entities: ExtractedEntitiesResponse
    recommended_filters: list[FilterRecommendationResponseItem]
    results: list[ProductSearchResultResponse]

class ExtractedEntitiesResponse(BaseModel):
    """Extracted query entities in API response."""

    brand: str | None = None
    category: str | None = None
    subcategory: str | None = None
    product_type: str | None = None
    color: str | None = None
    size: str | None = None
    gender: str | None = None
    condition: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    price_intent: str | None = None


class FilterRecommendationRequest(BaseModel):
    """Request body for filter recommendation."""

    query: str = Field(..., min_length=1)


class FilterRecommendationResponseItem(BaseModel):
    """Single recommended filter response item."""

    name: str
    value: str | float
    confidence: float
    source: str


class FilterRecommendationResponseModel(BaseModel):
    """Filter recommendation API response."""

    query: str
    normalized_query: str
    filters: list[FilterRecommendationResponseItem]