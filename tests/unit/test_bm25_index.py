from __future__ import annotations

import pandas as pd

from querysense.retrieval.bm25_index import BM25ProductIndex


def _build_products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Sony WH-1000XM4 Wireless Headphones",
                "description": "Black wireless noise cancelling headphones",
                "brand": "Sony",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 249.99,
                "currency": "EUR",
            },
            {
                "product_id": "p002",
                "title": "Nike Air Zoom Pegasus",
                "description": "Comfortable black running shoes for men",
                "brand": "Nike",
                "category": "Fashion",
                "subcategory": "Shoes",
                "color": "Black",
                "size": "44",
                "gender": "men",
                "condition": "new",
                "price": 89.99,
                "currency": "EUR",
            },
            {
                "product_id": "p003",
                "title": "IKEA Office Desk",
                "description": "White desk for home office work",
                "brand": "IKEA",
                "category": "Home",
                "subcategory": "Furniture",
                "color": "White",
                "size": "120x60",
                "gender": "unisex",
                "condition": "new",
                "price": 129.99,
                "currency": "EUR",
            },
        ]
    )


def test_bm25_index_search_returns_relevant_product_first() -> None:
    index = BM25ProductIndex(_build_products_df())

    result = index.search("wireless noise cancelling", top_k=3)

    assert result.iloc[0]["product_id"] == "p001"
    assert result.iloc[0]["bm25_score"] > 0


def test_bm25_index_search_returns_empty_for_unknown_query() -> None:
    index = BM25ProductIndex(_build_products_df())

    result = index.search("completely unrelated query", top_k=3)

    assert result.empty


def test_bm25_index_respects_top_k() -> None:
    index = BM25ProductIndex(_build_products_df())

    result = index.search("black", top_k=1)

    assert len(result) == 1
    assert result.iloc[0]["bm25_score"] > 0


def test_bm25_index_adds_search_text_column() -> None:
    index = BM25ProductIndex(_build_products_df())

    assert "search_text" in index.products_df.columns
    assert "sony wh-1000xm4 wireless headphones" in index.products_df.loc[0, "search_text"]