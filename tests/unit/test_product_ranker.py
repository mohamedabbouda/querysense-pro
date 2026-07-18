from __future__ import annotations

import pandas as pd

from querysense.query_understanding.entities import ExtractedEntities
from querysense.retrieval.product_ranker import rank_products


def _build_products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Sony WH-1000XM4 Wireless Headphones",
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
                "title": "Generic Black Headphones",
                "brand": "Generic",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 49.99,
                "currency": "EUR",
            },
            {
                "product_id": "p003",
                "title": "Sony Camera",
                "brand": "Sony",
                "category": "Electronics",
                "subcategory": "Cameras",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 399.99,
                "currency": "EUR",
            },
        ]
    )


def test_rank_products_puts_best_match_first() -> None:
    products_df = _build_products_df()
    entities = ExtractedEntities(
        brand="sony",
        subcategory="headphones",
        color="black",
        max_price=300,
    )

    result = rank_products(
        products_df=products_df,
        entities=entities,
        normalized_query="sony black headphones under 300",
    )

    assert result.iloc[0]["product_id"] == "p001"
    assert result.iloc[0]["score"] > result.iloc[1]["score"]
    assert "brand" in result.iloc[0]["match_reasons"]
    assert "subcategory" in result.iloc[0]["match_reasons"]
    assert "color" in result.iloc[0]["match_reasons"]
    assert "max_price" in result.iloc[0]["match_reasons"]


def test_rank_products_uses_title_token_matches() -> None:
    products_df = _build_products_df()
    entities = ExtractedEntities()

    result = rank_products(
        products_df=products_df,
        entities=entities,
        normalized_query="wireless headphones",
    )

    assert result.iloc[0]["product_id"] == "p001"
    assert "title:wireless" in result.iloc[0]["match_reasons"]
    assert "title:headphones" in result.iloc[0]["match_reasons"]

def test_rank_products_sorts_lower_price_when_scores_tie() -> None:
    products_df = _build_products_df().iloc[:2].copy()
    entities = ExtractedEntities(subcategory="headphones", color="black")

    result = rank_products(
        products_df=products_df,
        entities=entities,
        normalized_query="headphones",
    )

    assert result.iloc[0]["product_id"] == "p002"
    assert result.iloc[1]["product_id"] == "p001"
    assert result.iloc[0]["score"] == result.iloc[1]["score"]


def test_rank_products_handles_empty_dataframe() -> None:
    products_df = _build_products_df().iloc[:0].copy()
    entities = ExtractedEntities(brand="sony")

    result = rank_products(
        products_df=products_df,
        entities=entities,
        normalized_query="sony",
    )

    assert result.empty
    assert "score" in result.columns
    assert "match_reasons" in result.columns



def test_rank_products_adds_normalized_bm25_score() -> None:
    products_df = pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Generic Wireless Headphones",
                "brand": "Generic",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 99.99,
                "currency": "EUR",
                "bm25_score": 2.0,
            },
            {
                "product_id": "p002",
                "title": "Another Wireless Headphones",
                "brand": "Generic",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 89.99,
                "currency": "EUR",
                "bm25_score": 1.0,
            },
        ]
    )

    entities = ExtractedEntities(subcategory="headphones")

    ranked = rank_products(
        products_df=products_df,
        entities=entities,
        normalized_query="wireless headphones",
    )

    assert ranked.iloc[0]["product_id"] == "p001"
    assert ranked.iloc[0]["score"] > ranked.iloc[1]["score"]


def test_rank_products_adds_normalized_semantic_score() -> None:
    products_df = pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Generic Product",
                "brand": "Generic",
                "category": "Electronics",
                "subcategory": "Accessories",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 99.99,
                "currency": "EUR",
                "semantic_score": 0.9,
            },
            {
                "product_id": "p002",
                "title": "Another Product",
                "brand": "Generic",
                "category": "Electronics",
                "subcategory": "Accessories",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 89.99,
                "currency": "EUR",
                "semantic_score": 0.1,
            },
        ]
    )

    entities = ExtractedEntities(subcategory="accessories")

    ranked = rank_products(
        products_df=products_df,
        entities=entities,
        normalized_query="noise blocking headset",
    )

    assert ranked.iloc[0]["product_id"] == "p001"
    assert ranked.iloc[0]["score"] > ranked.iloc[1]["score"]