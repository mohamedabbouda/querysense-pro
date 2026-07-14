from __future__ import annotations

from pathlib import Path

import pandas as pd

from querysense.query_understanding.filter_service import (
    FilterRecommendationService,
    FilterRecommendationServiceConfig,
)


def _build_products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Nike Air Zoom Pegasus",
                "description": "Black running shoes for men",
                "brand": "Nike",
                "category": "Fashion",
                "subcategory": "Shoes",
                "color": "Black",
                "size": "44",
                "gender": "men",
                "condition": "new",
                "price": 89.99,
                "currency": "EUR",
                "attributes": "{}",
            },
            {
                "product_id": "p002",
                "title": "Sony WH-1000XM4 Wireless Headphones",
                "description": "Black wireless headphones",
                "brand": "Sony",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 249.99,
                "currency": "EUR",
                "attributes": "{}",
            },
            {
                "product_id": "p003",
                "title": "Dell XPS 13",
                "description": "Silver laptop",
                "brand": "Dell",
                "category": "Electronics",
                "subcategory": "Laptops",
                "color": "Silver",
                "size": "13-inch",
                "gender": "unisex",
                "condition": "new",
                "price": 999.99,
                "currency": "EUR",
                "attributes": "{}",
            },
        ]
    )


def _build_service(tmp_path: Path) -> FilterRecommendationService:
    products_path = tmp_path / "products.parquet"
    _build_products_df().to_parquet(products_path)

    return FilterRecommendationService(
        FilterRecommendationServiceConfig(
            products_path=products_path,
        )
    )


def test_filter_recommendation_service_recommends_filters(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    response = service.recommend("sony black headphones under 300")

    assert response.query == "sony black headphones under 300"
    assert response.normalized_query == "sony black headphones under 300"

    filters = {
        filter_.name: filter_.value
        for filter_ in response.filters
    }

    assert filters["brand"] == "sony"
    assert filters["category"] == "electronics"
    assert filters["subcategory"] == "headphones"
    assert filters["color"] == "black"
    assert filters["max_price"] == 300.0


def test_filter_recommendation_service_handles_typos(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    response = service.recommend("iphon blak")

    filters = {
        filter_.name: filter_.value
        for filter_ in response.filters
    }

    assert response.normalized_query == "iphone black"
    assert filters["brand"] == "apple"
    assert filters["color"] == "black"


def test_filter_recommendation_service_returns_empty_for_unknown_query(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    response = service.recommend("random unknown thing")

    assert response.query == "random unknown thing"
    assert response.filters == []