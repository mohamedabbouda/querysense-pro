from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from querysense.api.schemas import IntentPredictionRequest, IntentPredictionResponse
from querysense.config import load_project_configs
from querysense.query_understanding.intent_service import (
    IntentPredictionService,
    IntentServiceConfig,
)

app = FastAPI(
    title="QuerySense Pro API",
    description="API for query understanding and intent prediction.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_intent_service() -> IntentPredictionService:
    """Load and cache the intent prediction service."""
    configs = load_project_configs()
    base_config = configs["base"]
    training_config = configs["training"]
    intent_config = training_config["intent_classifier"]

    return IntentPredictionService(
        IntentServiceConfig(
            model_path=intent_config["model_output_path"],
            products_path=base_config["files"]["processed_products_parquet"],
        )
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict-intent", response_model=IntentPredictionResponse)
def predict_intent(request: IntentPredictionRequest) -> IntentPredictionResponse:
    """Predict the intent of a search query."""
    service = get_intent_service()
    prediction = service.predict(request.query)

    return IntentPredictionResponse(
        query=request.query,
        normalized_query=prediction.normalized_query,
        intent=prediction.intent,
        source=prediction.source,
        model_intent=prediction.model_intent,
    )