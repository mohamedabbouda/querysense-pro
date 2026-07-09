from __future__ import annotations

from querysense.config import load_project_configs
from querysense.data.load_products import load_products_csv, save_products_parquet


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]

    input_path = base_config["files"]["sample_products_csv"]
    output_path = base_config["files"]["processed_products_parquet"]

    df = load_products_csv(input_path)
    save_products_parquet(df, output_path)

    print(f"Loaded {len(df)} products from {input_path}")
    print(f"Saved processed products to {output_path}")


if __name__ == "__main__":
    main()