from __future__ import annotations

import pandas as pd

from querysense.query_understanding.entities import ExtractedEntities
from querysense.retrieval.product_filter import filter_products_by_entities


def _build_products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Nike Air Zoom Pegasus",
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
                "product_id": "p002",
                "title": "Adidas Ultraboost",
                "brand": "Adidas",
                "category": "Fashion",
                "subcategory": "Shoes",
                "color": "White",
                "size": "43",
                "gender": "women",
                "condition": "new",
                "price": 119.99,
                "currency": "EUR",
            },
            {
                "product_id": "p003",
                "title": "Sony WH-1000XM4",
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
        ]
    )


def test_filter_products_by_brand() -> None:
    products_df = _build_products_df()
    entities = ExtractedEntities(brand="sony")

    result = filter_products_by_entities(products_df, entities)

    assert result["product_id"].tolist() == ["p003"]


def test_filter_products_by_subcategory_and_color() -> None:
    products_df = _build_products_df()
    entities = ExtractedEntities(subcategory="shoes", color="black")

    result = filter_products_by_entities(products_df, entities)

    assert result["product_id"].tolist() == ["p001"]


def test_filter_products_by_max_price() -> None:
    products_df = _build_products_df()
    entities = ExtractedEntities(subcategory="shoes", max_price=100)

    result = filter_products_by_entities(products_df, entities)

    assert result["product_id"].tolist() == ["p001"]


def test_filter_products_by_min_price() -> None:
    products_df = _build_products_df()
    entities = ExtractedEntities(category="fashion", min_price=100)

    result = filter_products_by_entities(products_df, entities)

    assert result["product_id"].tolist() == ["p002"]


def test_filter_products_returns_all_when_no_entities() -> None:
    products_df = _build_products_df()
    entities = ExtractedEntities()

    result = filter_products_by_entities(products_df, entities)

    assert result["product_id"].tolist() == ["p001", "p002", "p003"]