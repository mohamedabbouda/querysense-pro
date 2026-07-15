from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor
from querysense.query_understanding.filter_recommendation import recommend_filters_from_entities
from querysense.query_understanding.intent_service import (
    IntentPredictionService,
    IntentServiceConfig,
)
from querysense.retrieval.bm25_index import BM25ProductIndex
from querysense.retrieval.product_filter import filter_products_by_entities
from querysense.retrieval.product_ranker import rank_products
from querysense.retrieval.search_result import ProductSearchResponse, ProductSearchResult


@dataclass(frozen=True)
class ProductSearchServiceConfig:
    """Configuration for product search service."""

    model_path: str | Path
    products_path: str | Path
    max_results: int = 10


class ProductSearchService:
    """End-to-end product search service."""

    def __init__(self, config: ProductSearchServiceConfig) -> None:
        self.config = config
        self.products_df = pd.read_parquet(config.products_path)
        self.bm25_index = BM25ProductIndex(self.products_df)

        self.entity_extractor = RuleBasedEntityExtractor.from_products(self.products_df)
        self.intent_service = IntentPredictionService(
            IntentServiceConfig(
                model_path=config.model_path,
                products_path=config.products_path,
            )
        )

    def search(self, query: str) -> ProductSearchResponse:
        """Search products for a user query."""
        intent_prediction = self.intent_service.predict(query)
        entities = self.entity_extractor.extract(query)

        filtered_products = filter_products_by_entities(
            products_df=self.products_df,
            entities=entities,
        )
        bm25_products = self.bm25_index.search(
            query= intent_prediction.normalized_query,
            top_k=self.config.max_results * 3,
        )
        candidate_products = _merge_candidate_products(
            filtered_products=filtered_products,
            bm25_products=bm25_products,
            )


        ranked_products = rank_products(
            products_df= candidate_products,
            entities=entities,
            normalized_query=intent_prediction.normalized_query,
        )
        
        top_products = ranked_products.head(self.config.max_results)

        results = [
            _row_to_search_result(row)
            for _, row in top_products.iterrows()
        ]
        recommended_filters = recommend_filters_from_entities(entities)


        return ProductSearchResponse(
            query=query,
            normalized_query=intent_prediction.normalized_query,
            intent=intent_prediction.intent,
            entities=entities,
            recommended_filters=recommended_filters,
            results=results,

        )


def _row_to_search_result(row: pd.Series) -> ProductSearchResult:
    return ProductSearchResult(
        product_id=str(row["product_id"]),
        title=str(row["title"]),
        brand=str(row["brand"]),
        category=str(row["category"]),
        subcategory=str(row["subcategory"]),
        color=str(row["color"]),
        size=str(row["size"]),
        gender=str(row["gender"]),
        condition=str(row["condition"]),
        price=float(row["price"]),
        currency=str(row["currency"]),
        score=float(row["score"]),
        match_reasons=list(row["match_reasons"]),
    )


def _merge_candidate_products(
    filtered_products: pd.DataFrame,
    bm25_products: pd.DataFrame,
    ) -> pd.DataFrame:
    """Merge structured-filter candidates and BM25 candidates."""
    if filtered_products.empty and bm25_products.empty:
        return filtered_products.copy()

    candidate_products = pd.concat(
        [filtered_products, bm25_products],
        ignore_index=True,
    )

    candidate_products = candidate_products.drop_duplicates(
        subset=["product_id"],
        keep="first",
    )

    return candidate_products.reset_index(drop=True)