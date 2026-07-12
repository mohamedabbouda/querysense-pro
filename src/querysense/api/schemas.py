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
    results: list[ProductSearchResultResponse]