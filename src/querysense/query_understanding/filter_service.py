from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor
from querysense.query_understanding.filter_recommendation import (
    FilterRecommendationResponse,
    recommend_filters_from_entities,
)
from querysense.query_understanding.normalizer import QueryNormalizer


@dataclass(frozen=True)
class FilterRecommendationServiceConfig:
    """Configuration for filter recommendation service."""

    products_path: str | Path


class FilterRecommendationService:
    """Reusable service for filter recommendation."""

    def __init__(self, config: FilterRecommendationServiceConfig) -> None:
        self.config = config

        products_df = pd.read_parquet(config.products_path)

        self.entity_extractor = RuleBasedEntityExtractor.from_products(products_df)
        self.normalizer = QueryNormalizer()

    def recommend(self, query: str) -> FilterRecommendationResponse:
        """Recommend filters for a single query."""
        normalized = self.normalizer.normalize(query)
        entities = self.entity_extractor.extract(query)
        filters = recommend_filters_from_entities(entities)

        return FilterRecommendationResponse(
            query=query,
            normalized_query=normalized.normalized_query,
            filters=filters,
        )