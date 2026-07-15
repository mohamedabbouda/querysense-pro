from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from querysense.api.schemas import (
    ExtractedEntitiesResponse,
    FilterRecommendationRequest,
    FilterRecommendationResponseItem,
    FilterRecommendationResponseModel,
    IntentPredictionRequest,
    IntentPredictionResponse,
    ProductSearchRequest,
    ProductSearchResponseModel,
    ProductSearchResultResponse,
)
from querysense.config import load_project_configs
from querysense.query_understanding.filter_service import (
    FilterRecommendationService,
    FilterRecommendationServiceConfig,
)
from querysense.query_understanding.intent_service import (
    IntentPredictionService,
    IntentServiceConfig,
)
from querysense.retrieval.search_service import ProductSearchService, ProductSearchServiceConfig

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


@lru_cache(maxsize=1)
def get_product_search_service() -> ProductSearchService:
    """Load and cache the product search service."""
    configs = load_project_configs()
    base_config = configs["base"]
    training_config = configs["training"]
    intent_config = training_config["intent_classifier"]

    return ProductSearchService(
        ProductSearchServiceConfig(
            model_path=intent_config["model_output_path"],
            products_path=base_config["files"]["processed_products_parquet"],
        )
    )

@lru_cache(maxsize=1)
def get_filter_recommendation_service() -> FilterRecommendationService:
    """Load and cache the filter recommendation service."""
    configs = load_project_configs()
    base_config = configs["base"]

    return FilterRecommendationService(
        FilterRecommendationServiceConfig(
            products_path=base_config["files"]["processed_products_parquet"],
        )
    )


@app.post("/search", response_model=ProductSearchResponseModel)
def search_products(request: ProductSearchRequest) -> ProductSearchResponseModel:
    """Search products for a user query."""
    service = get_product_search_service()
    response = service.search(request.query)

    return ProductSearchResponseModel(
    query=response.query,
    normalized_query=response.normalized_query,
    intent=response.intent,
    entities=ExtractedEntitiesResponse(
        brand=response.entities.brand,
        category=response.entities.category,
        subcategory=response.entities.subcategory,
        product_type=response.entities.product_type,
        color=response.entities.color,
        size=response.entities.size,
        gender=response.entities.gender,
        condition=response.entities.condition,
        min_price=response.entities.min_price,
        max_price=response.entities.max_price,
        price_intent=response.entities.price_intent,
    ),
    recommended_filters=[
        FilterRecommendationResponseItem(
            name=filter_.name,
            value=filter_.value,
            confidence=filter_.confidence,
            source=filter_.source,
        )
        for filter_ in response.recommended_filters
    ],
    results=[
        ProductSearchResultResponse(
            product_id=result.product_id,
            title=result.title,
            brand=result.brand,
            category=result.category,
            subcategory=result.subcategory,
            color=result.color,
            size=result.size,
            gender=result.gender,
            condition=result.condition,
            price=result.price,
            currency=result.currency,
            score=result.score,
            bm25_score=result.bm25_score,
            match_reasons=result.match_reasons,
        )
        for result in response.results
    ],
    )


@app.post("/recommend-filters", response_model=FilterRecommendationResponseModel)
def recommend_filters(
    request: FilterRecommendationRequest,
) -> FilterRecommendationResponseModel:
    """Recommend product filters for a user query."""
    service = get_filter_recommendation_service()
    response = service.recommend(request.query)

    return FilterRecommendationResponseModel(
        query=response.query,
        normalized_query=response.normalized_query,
        filters=[
            FilterRecommendationResponseItem(
                name=filter_.name,
                value=filter_.value,
                confidence=filter_.confidence,
                source=filter_.source,
            )
            for filter_ in response.filters
        ],
    )



