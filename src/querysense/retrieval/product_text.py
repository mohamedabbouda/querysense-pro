from __future__ import annotations

import pandas as pd

from querysense.utils.text import normalize_text_basic

DEFAULT_PRODUCT_TEXT_COLUMNS = [
    "title",
    "description",
    "brand",
    "category",
    "subcategory",
    "color",
    "size",
    "gender",
    "condition",
]


def build_product_search_text(
    product: pd.Series,
    columns: list[str] | None = None,
) -> str:
    """Build normalized searchable text for one product."""
    selected_columns = columns or DEFAULT_PRODUCT_TEXT_COLUMNS

    values: list[str] = []

    for column in selected_columns:
        if column not in product.index:
            continue

        value = product[column]

        if pd.isna(value):
            continue

        normalized_value = normalize_text_basic(str(value))

        if normalized_value:
            values.append(normalized_value)

    return " ".join(values)


def add_product_search_text_column(
    products_df: pd.DataFrame,
    output_column: str = "search_text",
) -> pd.DataFrame:
    """Add normalized searchable text column to a product catalog."""
    products_with_text = products_df.copy()

    products_with_text[output_column] = products_with_text.apply(
        build_product_search_text,
        axis=1,
    )

    return products_with_text