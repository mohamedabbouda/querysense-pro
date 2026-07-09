from __future__ import annotations

import pandas as pd

from querysense.config import load_project_configs
from querysense.data.generate_queries import generate_synthetic_queries
from querysense.data.query_dataset import add_normalized_query_columns


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]

    products_path = base_config["files"]["processed_products_parquet"]
    output_path = base_config["files"]["synthetic_queries_csv"]

    products_df = pd.read_parquet(products_path)
    queries_df = generate_synthetic_queries(products_df)
    queries_df = add_normalized_query_columns(queries_df)

    queries_df.to_csv(output_path, index=False)

    print(f"Loaded {len(products_df)} products from {products_path}")
    print(f"Generated {len(queries_df)} synthetic queries")
    print(f"Saved synthetic queries to {output_path}")
    print()
    print(queries_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()