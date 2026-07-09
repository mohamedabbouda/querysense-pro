from __future__ import annotations

import json

import pandas as pd

from querysense.data.generate_queries import generate_synthetic_queries
from querysense.data.query_dataset import add_normalized_query_columns


def test_generate_synthetic_queries_shape() -> None:
    products_df = pd.read_parquet("data/processed/products.parquet")
    queries_df = generate_synthetic_queries(products_df, shuffle=False)

    assert queries_df.shape == (140, 11)


def test_generate_synthetic_queries_contains_expected_intents() -> None:
    products_df = pd.read_parquet("data/processed/products.parquet")
    queries_df = generate_synthetic_queries(products_df, shuffle=False)

    expected_intents = {
        "product_search",
        "filtered_product_search",
        "brand_search",
        "price_search",
        "category_search",
    }

    assert set(queries_df["intent"].unique()) == expected_intents


def test_generate_synthetic_queries_has_target_product_ids() -> None:
    products_df = pd.read_parquet("data/processed/products.parquet")
    queries_df = generate_synthetic_queries(products_df, shuffle=False)

    assert queries_df["target_product_id"].notna().all()
    assert queries_df["target_product_id"].str.startswith("p").all()


def test_first_product_generates_expected_queries() -> None:
    products_df = pd.read_parquet("data/processed/products.parquet")
    queries_df = generate_synthetic_queries(products_df, shuffle=False)

    first_product_queries = queries_df[queries_df["target_product_id"] == "p001"]["query"].tolist()

    assert "nike black shoes" in first_product_queries
    assert "nike shoes 44" in first_product_queries
    assert "black shoes for men" in first_product_queries


def test_add_normalized_query_columns() -> None:
    queries_df = pd.DataFrame(
        [
            {
                "query": "NIKE!!! black   shoez",
                "intent": "product_search",
                "target_product_id": "p001",
                "brand": "nike",
                "category": "fashion",
                "subcategory": "shoes",
                "color": "black",
                "size": "",
                "gender": "",
                "condition": "",
                "max_price": None,
            }
        ]
    )

    enriched_df = add_normalized_query_columns(queries_df)

    assert enriched_df.loc[0, "normalized_query"] == "nike black shoes"
    assert json.loads(enriched_df.loc[0, "tokens"]) == ["nike", "black", "shoez"]
    assert json.loads(enriched_df.loc[0, "corrected_tokens"]) == ["nike", "black", "shoes"]
    assert json.loads(enriched_df.loc[0, "corrections"]) == {"shoez": "shoes"}

    
def test_price_query_label_matches_query_text() -> None:
    products_df = pd.read_parquet("data/processed/products.parquet")
    queries_df = generate_synthetic_queries(products_df, shuffle=False)

    price_query = queries_df[
        (queries_df["target_product_id"] == "p001")
        & (queries_df["intent"] == "price_search")
    ].iloc[0]

    assert price_query["query"] == "nike shoes under 139"
    assert price_query["max_price"] == 139.0    