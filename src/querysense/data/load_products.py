from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from querysense.data.product_schema import REQUIRED_PRODUCT_COLUMNS, Product


class ProductDataError(ValueError):
    """Raised when product data is missing required structure or has invalid values."""


def validate_required_columns(df: pd.DataFrame) -> None:
    """Check whether the DataFrame contains all required product columns."""
    missing_columns = sorted(set(REQUIRED_PRODUCT_COLUMNS) - set(df.columns))

    if missing_columns:
        raise ProductDataError(f"Missing required columns: {missing_columns}")


def validate_product_records(df: pd.DataFrame) -> list[Product]:
    """Validate each product row using the Product schema."""
    products: list[Product] = []
    errors: list[str] = []

    for row_index, row in df.iterrows():
        try:
            product = Product.model_validate(row.to_dict())
            products.append(product)
        except ValidationError as exc:
            errors.append(f"Row {row_index}: {exc}")

    if errors:
        error_preview = "\n\n".join(errors[:5])
        raise ProductDataError(f"Invalid product records found:\n{error_preview}")

    return products


def load_products_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate products from a CSV file."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Product CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_required_columns(df)
    validate_product_records(df)

    return df


def save_products_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save products to Parquet format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)