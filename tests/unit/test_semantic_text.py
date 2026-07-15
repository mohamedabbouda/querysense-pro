from __future__ import annotations

import pandas as pd

from querysense.retrieval.semantic_text import (
    add_product_semantic_text_column,
    build_product_semantic_text,
)


def test_build_product_semantic_text_includes_natural_product_fields() -> None:
    product = pd.Series(
        {
            "title": "Sony WH-1000XM4 Wireless Headphones",
            "description": "Black wireless noise cancelling headphones",
            "brand": "Sony",
            "category": "Electronics",
            "subcategory": "Headphones",
            "color": "Black",
            "size": "one-size",
            "gender": "unisex",
            "condition": "new",
        }
    )

    semantic_text = build_product_semantic_text(product)

    assert "sony wh-1000xm4 wireless headphones" in semantic_text
    assert "black wireless noise cancelling headphones" in semantic_text
    assert "brand sony" in semantic_text
    assert "category electronics" in semantic_text
    assert "subcategory headphones" in semantic_text


def test_build_product_semantic_text_skips_missing_values() -> None:
    product = pd.Series(
        {
            "title": "Nike Running Shoes",
            "description": None,
            "brand": "Nike",
            "category": "Fashion",
        }
    )

    semantic_text = build_product_semantic_text(product)

    assert "nike running shoes" in semantic_text
    assert "brand nike" in semantic_text
    assert "none" not in semantic_text


def test_add_product_semantic_text_column_adds_column_without_mutating_input() -> None:
    products_df = pd.DataFrame(
        [
            {
                "title": "Office Desk",
                "description": "White desk for home office",
                "brand": "IKEA",
                "category": "Home",
                "subcategory": "Furniture",
            }
        ]
    )

    result = add_product_semantic_text_column(products_df)

    assert "semantic_text" in result.columns
    assert "semantic_text" not in products_df.columns
    assert "office desk" in result.loc[0, "semantic_text"]