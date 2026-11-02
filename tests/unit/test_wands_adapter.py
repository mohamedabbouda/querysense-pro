
from __future__ import annotations

import pandas as pd
import pytest

from querysense.data.wands_adapter import (
    WandsDataError,
    adapt_wands_products,
    load_wands_products,
)


def make_wands_products() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "123",
                "product_name": "Blue Velvet Sofa",
                "product_class": "Sofas",
                "category_hierarchy": "Furniture / Living Room / Sofas",
                "product_description": "Three-seat sofa",
                "product_features": "Color:Blue|Material:Velvet",
                "rating_count": 50,
            }
        ]
    )


def test_adapter_maps_product_fields() -> None:
    adapted = adapt_wands_products(make_wands_products())

    product = adapted.iloc[0]

    assert product["product_id"] == "123"
    assert product["title"] == "Blue Velvet Sofa"
    assert product["category"] == "Furniture"
    assert product["subcategory"] == "Sofas"
    assert product["description"] == "Three-seat sofa"
    assert product["attributes"] == "Color:Blue|Material:Velvet"
    assert product["rating_count"] == 50


def test_adapter_preserves_missing_information() -> None:
    products = make_wands_products().drop(
        columns=["product_description", "product_features"]
    )

    adapted = adapt_wands_products(products)

    assert pd.isna(adapted.iloc[0]["description"])
    assert pd.isna(adapted.iloc[0]["attributes"])
    assert "price" not in adapted.columns
    assert "brand" not in adapted.columns


def test_adapter_rejects_missing_columns() -> None:
    products = make_wands_products().drop(columns=["product_name"])

    with pytest.raises(WandsDataError, match="Missing WANDS columns"):
        adapt_wands_products(products)


def test_adapter_rejects_duplicate_ids() -> None:
    products = pd.concat(
        [make_wands_products(), make_wands_products()],
        ignore_index=True,
    )

    with pytest.raises(WandsDataError, match="Duplicate product IDs"):
        adapt_wands_products(products)

def test_adapter_loads_csv(tmp_path) -> None:
    path = tmp_path / "product.csv"

    products = make_wands_products().rename(
        columns={"category_hierarchy": "category hierarchy"}
    )
    products.to_csv(path, sep="\t", index=False)

    adapted = load_wands_products(path)

    assert len(adapted) == 1
    assert adapted.iloc[0]["product_id"] == "123"

def test_loader_handles_real_wands_tsv_format(tmp_path) -> None:
    path = tmp_path / "product.csv"

    products = make_wands_products().rename(
        columns={"category_hierarchy": "category hierarchy"}
    )

    products.to_csv(path, sep="\t", index=False)

    adapted = load_wands_products(path)

    assert len(adapted) == 1
    assert adapted.iloc[0]["product_id"] == "123"
    assert adapted.iloc[0]["category"] == "Furniture"