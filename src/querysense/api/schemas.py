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