from __future__ import annotations

import pandas as pd

from querysense.config import load_project_configs
from querysense.data.entity_dataset import add_extracted_entity_columns
from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]

    products_path = base_config["files"]["processed_products_parquet"]
    queries_path = base_config["files"]["synthetic_queries_csv"]
    output_path = base_config["files"]["query_understanding_dataset_csv"]

    products_df = pd.read_parquet(products_path)
    queries_df = pd.read_csv(queries_path)

    extractor = RuleBasedEntityExtractor.from_products(products_df)
    enriched_df = add_extracted_entity_columns(queries_df, extractor)

    enriched_df.to_csv(output_path, index=False)

    print(f"Loaded {len(products_df)} products from {products_path}")
    print(f"Loaded {len(queries_df)} queries from {queries_path}")
    print(f"Saved query understanding dataset to {output_path}")
    print()
    print(
        enriched_df[
            [
                "query",
                "intent",
                "brand",
                "category",
                "subcategory",
                "color",
                "size",
                "gender",
                "condition",
                "max_price",
                "extracted_brand",
                "extracted_category",
                "extracted_subcategory",
                "extracted_color",
                "extracted_size",
                "extracted_gender",
                "extracted_condition",
                "extracted_max_price",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()