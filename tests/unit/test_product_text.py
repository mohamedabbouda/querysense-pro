from __future__ import annotations

import pandas as pd

from querysense.retrieval.product_text import (
    add_product_search_text_column,
    build_product_search_text,
)


def test_build_product_search_text() -> None:
    product = pd.Series(
        {
            "title": "Sony WH-1000XM4 Wireless Headphones",
            "description": "Black noise cancelling headphones",
            "brand": "Sony",
            "category": "Electronics",
            "subcategory": "Headphones",
            "color": "Black",
            "size": "one-size",
            "gender": "unisex",
            "condition": "new",
        }
    )

    search_text = build_product_search_text(product)

    assert "sony wh-1000xm4 wireless headphones" in search_text
    assert "black noise cancelling headphones" in search_text
    assert "electronics" in search_text
    assert search_text == search_text.lower()


def test_build_product_search_text_skips_missing_columns() -> None:
    product = pd.Series(
        {
            "title": "Nike Shoes",
            "brand": "Nike",
        }
    )

    search_text = build_product_search_text(product)

    assert search_text == "nike shoes nike"


def test_add_product_search_text_column() -> None:
    products_df = pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Sony Headphones",
                "description": "Wireless headphones",
                "brand": "Sony",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
            }
        ]
    )

    result = add_product_search_text_column(products_df)

    assert "search_text" in result.columns
    assert result.loc[0, "search_text"].startswith("sony headphones")
    assert "wireless headphones" in result.loc[0, "search_text"]