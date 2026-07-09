from __future__ import annotations

import pandas as pd

from querysense.data.entity_dataset import add_extracted_entity_columns
from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor


def test_add_extracted_entity_columns() -> None:
    products_df = pd.read_parquet("data/processed/products.parquet")
    extractor = RuleBasedEntityExtractor.from_products(products_df)

    queries_df = pd.DataFrame(
        [
            {
                "query": "nike black shoes size 44",
                "intent": "filtered_product_search",
                "brand": "nike",
                "category": "fashion",
                "subcategory": "shoes",
                "color": "black",
                "size": "44",
                "gender": "",
                "condition": "",
                "max_price": None,
            }
        ]
    )

    enriched_df = add_extracted_entity_columns(queries_df, extractor)

    assert "extracted_brand" in enriched_df.columns
    assert "extracted_color" in enriched_df.columns
    assert "extracted_size" in enriched_df.columns
    assert "extracted_max_price" in enriched_df.columns

    assert enriched_df.loc[0, "extracted_brand"] == "nike"
    assert enriched_df.loc[0, "extracted_color"] == "black"
    assert enriched_df.loc[0, "extracted_size"] == "44"