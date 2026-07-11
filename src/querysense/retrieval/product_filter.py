from __future__ import annotations

import pandas as pd

from querysense.query_understanding.entities import ExtractedEntities
from querysense.utils.text import normalize_text_basic


def filter_products_by_entities(
    products_df: pd.DataFrame,
    entities: ExtractedEntities,
) -> pd.DataFrame:
    """Filter product catalog using extracted query entities."""
    filtered_df = products_df.copy()

    filtered_df = _filter_text_column(filtered_df, "brand", entities.brand)
    filtered_df = _filter_text_column(filtered_df, "category", entities.category)
    filtered_df = _filter_text_column(filtered_df, "subcategory", entities.subcategory)
    filtered_df = _filter_text_column(filtered_df, "color", entities.color)
    filtered_df = _filter_text_column(filtered_df, "size", entities.size)
    filtered_df = _filter_text_column(filtered_df, "gender", entities.gender)
    filtered_df = _filter_text_column(filtered_df, "condition", entities.condition)

    if entities.min_price is not None:
        filtered_df = filtered_df[filtered_df["price"] >= entities.min_price]

    if entities.max_price is not None:
        filtered_df = filtered_df[filtered_df["price"] <= entities.max_price]

    return filtered_df.reset_index(drop=True)


def _filter_text_column(
    products_df: pd.DataFrame,
    column_name: str,
    expected_value: str | None,
) -> pd.DataFrame:
    if expected_value is None:
        return products_df

    normalized_expected = normalize_text_basic(expected_value)

    normalized_column = products_df[column_name].astype(str).map(normalize_text_basic)

    return products_df[normalized_column == normalized_expected]