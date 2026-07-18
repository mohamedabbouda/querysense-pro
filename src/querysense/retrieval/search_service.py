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
from querysense.retrieval.semantic_index import SemanticProductIndex, TextEmbeddingModel


@dataclass(frozen=True)
class ProductSearchServiceConfig:
    """Configuration for product search service."""

    model_path: str | Path
    products_path: str | Path
    max_results: int = 10
    use_semantic_search: bool = False


class ProductSearchService:
    """End-to-end product search service."""

    def __init__(
    self,
    config: ProductSearchServiceConfig,
    embedding_model: TextEmbeddingModel | None = None,
    ) -> None:
        self.config = config
        self.products_df = pd.read_parquet(config.products_path)
        self.bm25_index = BM25ProductIndex(self.products_df)
        self.semantic_index: SemanticProductIndex | None = None
        if config.use_semantic_search:
            if embedding_model is None:
                self.semantic_index = SemanticProductIndex.from_model_name(
                    products_df=self.products_df,
                    )
            else:
                self.semantic_index = SemanticProductIndex(
                    products_df=self.products_df,
                    embedding_model=embedding_model,
                    )

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
            query=intent_prediction.normalized_query,
            top_k=self.config.max_results * 3,
            )
        semantic_products = pd.DataFrame()
        if self.semantic_index is not None:
            semantic_products = self.semantic_index.search(
                query=intent_prediction.normalized_query,
                top_k=self.config.max_results * 3,
                )
        candidate_products = _merge_candidate_products(
            filtered_products=filtered_products,
            bm25_products=bm25_products,
            semantic_products=semantic_products,
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
    bm25_score = row.get("bm25_score", 0.0)
    safe_bm25_score = 0.0 if pd.isna(bm25_score) else float(bm25_score)
    semantic_score = row.get("semantic_score", 0.0)
    safe_semantic_score = 0.0 if pd.isna(semantic_score) else float(semantic_score)

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
        bm25_score=safe_bm25_score,
        semantic_score=safe_semantic_score,
        match_reasons=list(row["match_reasons"]),
        
    )


def _merge_candidate_products(
    filtered_products: pd.DataFrame,
    bm25_products: pd.DataFrame,
    semantic_products: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge structured, BM25, and semantic candidates.

    If a product appears in multiple sources, preserve retrieval scores.
    """
    if semantic_products is None:
        semantic_products = pd.DataFrame()

    if filtered_products.empty and bm25_products.empty and semantic_products.empty:
        return filtered_products.copy()

    filtered_candidates = filtered_products.copy()
    bm25_candidates = bm25_products.copy()
    semantic_candidates = semantic_products.copy()

    for candidates in [filtered_candidates, bm25_candidates, semantic_candidates]:
        if "bm25_score" not in candidates.columns:
            candidates["bm25_score"] = 0.0

        if "semantic_score" not in candidates.columns:
            candidates["semantic_score"] = 0.0

        candidates["bm25_score"] = candidates["bm25_score"].fillna(0.0)
        candidates["semantic_score"] = candidates["semantic_score"].fillna(0.0)

    candidate_products = pd.concat(
        [filtered_candidates, bm25_candidates, semantic_candidates],
        ignore_index=True,
    )

    candidate_products["bm25_score"] = candidate_products["bm25_score"].fillna(0.0)
    candidate_products["semantic_score"] = candidate_products["semantic_score"].fillna(0.0)

    candidate_products = (
        candidate_products.groupby("product_id", as_index=False)
        .agg(_aggregate_candidate_group)
        .reset_index(drop=True)
    )

    return candidate_products


def _aggregate_candidate_group(group: pd.Series) -> object:
    """Aggregate duplicate product candidates."""
    if group.name == "bm25_score":
        return float(group.max())

    if group.name == "semantic_score":
        return float(group.max())

    return group.iloc[0]