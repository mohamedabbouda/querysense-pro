from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from querysense.evaluation.search_relevance import SearchRelevanceExample
from querysense.query_understanding.entities import ExtractedEntities
from querysense.retrieval.search_result import ProductSearchResponse, ProductSearchResult
from querysense.retrieval.search_service import ProductSearchService
from querysense.utils.text import normalize_text_basic


@dataclass(frozen=True)
class LTRFeatureRow:
    """One query-product training row for a learning-to-rank model."""

    query: str
    normalized_query: str
    expanded_query: str
    product_id: str
    label: int
    manual_score: float
    bm25_score: float
    semantic_score: float
    price: float
    brand_match: int
    category_match: int
    subcategory_match: int
    color_match: int
    size_match: int
    gender_match: int
    condition_match: int
    has_bm25_match: int
    has_semantic_match: int
    title_token_overlap_count: int
    title_token_overlap_ratio: float


def build_ltr_feature_dataset(
    search_service: ProductSearchService,
    relevance_examples: list[SearchRelevanceExample],
) -> pd.DataFrame:
    """Build a pointwise learning-to-rank feature dataset.

    Each row represents one query-product candidate pair.
    """
    rows: list[LTRFeatureRow] = []

    for example in relevance_examples:
        response = search_service.search(example.query)

        for result in response.results:
            label = int(result.product_id in example.relevant_product_ids)
            rows.append(
                build_ltr_feature_row(
                    response=response,
                    result=result,
                    label=label,
                )
            )

    return pd.DataFrame([row.__dict__ for row in rows])


def build_ltr_feature_row(
    response: ProductSearchResponse,
    result: ProductSearchResult,
    label: int,
) -> LTRFeatureRow:
    """Build one LTR feature row from a search response and product result."""
    query_tokens = set(normalize_text_basic(response.normalized_query).split())
    title_tokens = set(normalize_text_basic(result.title).split())

    title_overlap_count = len(query_tokens & title_tokens)
    title_overlap_ratio = _safe_ratio(
        numerator=title_overlap_count,
        denominator=len(query_tokens),
    )

    return LTRFeatureRow(
        query=response.query,
        normalized_query=response.normalized_query,
        expanded_query=response.expanded_query,
        product_id=result.product_id,
        label=label,
        manual_score=result.score,
        bm25_score=result.bm25_score,
        semantic_score=result.semantic_score,
        price=result.price,
        brand_match=_entity_matches(response.entities, "brand", result.brand),
        category_match=_entity_matches(response.entities, "category", result.category),
        subcategory_match=_entity_matches(
            response.entities,
            "subcategory",
            result.subcategory,
        ),
        color_match=_entity_matches(response.entities, "color", result.color),
        size_match=_entity_matches(response.entities, "size", result.size),
        gender_match=_entity_matches(response.entities, "gender", result.gender),
        condition_match=_entity_matches(
            response.entities,
            "condition",
            result.condition,
        ),
        has_bm25_match=int(result.bm25_score > 0),
        has_semantic_match=int(result.semantic_score > 0),
        title_token_overlap_count=title_overlap_count,
        title_token_overlap_ratio=title_overlap_ratio,
    )


def _entity_matches(
    entities: ExtractedEntities,
    entity_name: str,
    product_value: str,
) -> int:
    expected_value = getattr(entities, entity_name)

    if expected_value is None:
        return 0

    return int(
        normalize_text_basic(str(expected_value))
        == normalize_text_basic(str(product_value))
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator