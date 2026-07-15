from __future__ import annotations

import pandas as pd

from querysense.query_understanding.entities import ExtractedEntities
from querysense.utils.text import normalize_text_basic

MATCH_WEIGHTS = {
    "brand": 3.0,
    "subcategory": 2.5,
    "category": 1.5,
    "color": 1.0,
    "size": 1.0,
    "gender": 1.0,
    "condition": 1.0,
    "min_price": 0.5,
    "max_price": 0.5,
    "title_token": 0.25,
    "BM25_SCORE_WEIGHT" : 1.0,
}


def rank_products(
    products_df: pd.DataFrame,
    entities: ExtractedEntities,
    normalized_query: str,
) -> pd.DataFrame:
    """Add ranking score and match reasons to products."""
    if products_df.empty:
        ranked_df = products_df.copy()
        ranked_df["score"] = []
        ranked_df["match_reasons"] = []
        return ranked_df

    ranked_df = products_df.copy()


    scores: list[float] = []
    match_reasons: list[list[str]] = []

    query_tokens = set(normalized_query.split())

    for _, product in ranked_df.iterrows():
        score, reasons = _score_product(
            product=product,
            entities=entities,
            query_tokens=query_tokens,
        )
        scores.append(score)
        match_reasons.append(reasons)

    ranked_df["score"] = scores
    ranked_df["match_reasons"] = match_reasons
    if "bm25_score" in ranked_df.columns:
        ranked_df["score"] = ranked_df["score"] + (
        _normalize_bm25_scores(ranked_df["bm25_score"]) * MATCH_WEIGHTS["BM25_SCORE_WEIGHT"] 
        )

    return ranked_df.sort_values(
        by=["score", "price"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _score_product(
    product: pd.Series,
    entities: ExtractedEntities,
    query_tokens: set[str],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    

    score, reasons = _score_text_match(
        product=product,
        column_name="brand",
        expected_value=entities.brand,
        score=score,
        reasons=reasons,
    )
    score, reasons = _score_text_match(
        product=product,
        column_name="subcategory",
        expected_value=entities.subcategory,
        score=score,
        reasons=reasons,
    )
    score, reasons = _score_text_match(
        product=product,
        column_name="category",
        expected_value=entities.category,
        score=score,
        reasons=reasons,
    )
    score, reasons = _score_text_match(
        product=product,
        column_name="color",
        expected_value=entities.color,
        score=score,
        reasons=reasons,
    )
    score, reasons = _score_text_match(
        product=product,
        column_name="size",
        expected_value=entities.size,
        score=score,
        reasons=reasons,
    )
    score, reasons = _score_text_match(
        product=product,
        column_name="gender",
        expected_value=entities.gender,
        score=score,
        reasons=reasons,
    )
    score, reasons = _score_text_match(
        product=product,
        column_name="condition",
        expected_value=entities.condition,
        score=score,
        reasons=reasons,
    )

    price = float(product["price"])

    if entities.min_price is not None and price >= entities.min_price:
        score += MATCH_WEIGHTS["min_price"]
        reasons.append("min_price")

    if entities.max_price is not None and price <= entities.max_price:
        score += MATCH_WEIGHTS["max_price"]
        reasons.append("max_price")

    title_tokens = set(normalize_text_basic(str(product["title"])).split())
    matching_title_tokens = sorted(query_tokens & title_tokens)

    if matching_title_tokens:
        score += len(matching_title_tokens) * MATCH_WEIGHTS["title_token"]
        reasons.extend([f"title:{token}" for token in matching_title_tokens])
    
    bm25_score = product.get("bm25_score", 0.0)
    if pd.notna(bm25_score) and float(bm25_score) > 0:
        reasons.append("bm25")
        

    return score, reasons


def _score_text_match(
    product: pd.Series,
    column_name: str,
    expected_value: str | None,
    score: float,
    reasons: list[str],
) -> tuple[float, list[str]]:
    if expected_value is None:
        return score, reasons

    actual_value = normalize_text_basic(str(product[column_name]))
    normalized_expected = normalize_text_basic(expected_value)

    if actual_value == normalized_expected:
        score += MATCH_WEIGHTS[column_name]
        reasons.append(column_name)

    return score, reasons


def _normalize_bm25_scores(scores: pd.Series) -> pd.Series:
    """Normalize BM25 scores to a 0-1 range."""
    safe_scores = scores.fillna(0.0).astype(float)

    max_score = safe_scores.max()

    if max_score <= 0:
        return safe_scores * 0.0

    return safe_scores / max_score
