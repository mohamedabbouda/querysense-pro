from __future__ import annotations

import pandas as pd

from querysense.utils.text import normalize_text_basic


def build_product_semantic_text(product: pd.Series) -> str:
    """Build natural product text for semantic retrieval."""
    title = _get_value(product, "title")
    description = _get_value(product, "description")
    brand = _get_value(product, "brand")
    category = _get_value(product, "category")
    subcategory = _get_value(product, "subcategory")
    color = _get_value(product, "color")
    size = _get_value(product, "size")
    gender = _get_value(product, "gender")
    condition = _get_value(product, "condition")

    parts = [
        title,
        description,
        f"Brand: {brand}" if brand else "",
        f"Category: {category}" if category else "",
        f"Subcategory: {subcategory}" if subcategory else "",
        f"Color: {color}" if color else "",
        f"Size: {size}" if size else "",
        f"Gender: {gender}" if gender else "",
        f"Condition: {condition}" if condition else "",
    ]

    semantic_text = ". ".join(part for part in parts if part)

    return normalize_text_basic(semantic_text)


def add_product_semantic_text_column(
    products_df: pd.DataFrame,
    output_column: str = "semantic_text",
) -> pd.DataFrame:
    """Add semantic text column to a product dataframe."""
    products_with_text = products_df.copy()

    products_with_text[output_column] = products_with_text.apply(
        build_product_semantic_text,
        axis=1,
    )

    return products_with_text


def _get_value(product: pd.Series, column_name: str) -> str:
    if column_name not in product.index:
        return ""

    value = product[column_name]

    if pd.isna(value):
        return ""

    return str(value).strip()