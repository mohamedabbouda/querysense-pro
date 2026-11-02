
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_WANDS_COLUMNS = {
    "product_id",
    "product_name",
    "product_class",
    "category_hierarchy",
}

OPTIONAL_WANDS_COLUMNS = {
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
}


class WandsDataError(ValueError):
    """Raised when WANDS product records are invalid."""
    



def adapt_wands_products(products_df: pd.DataFrame) -> pd.DataFrame:
    """Adapt WANDS products without inventing missing attributes."""
    missing_columns = REQUIRED_WANDS_COLUMNS - set(products_df.columns)

    if missing_columns:
        raise WandsDataError(
            f"Missing WANDS columns: {sorted(missing_columns)}"
        )

    adapted = products_df.copy()

    if adapted["product_id"].isna().any():
        raise WandsDataError("Product IDs must not be missing")

    adapted["product_id"] = (
        adapted["product_id"].astype("string").str.strip()
    )

    adapted["title"] = (
        adapted["product_name"].astype("string").str.strip()
    )

    if adapted["product_id"].eq("").any():
        raise WandsDataError("Product IDs must not be empty")

    if adapted["title"].isna().any() or adapted["title"].eq("").any():
        raise WandsDataError("Product titles must not be missing or empty")

    if adapted["product_id"].duplicated().any():
        raise WandsDataError("Duplicate product IDs detected")

    adapted["subcategory"] = (
        adapted["product_class"].astype("string").str.strip()
    )

    adapted["category"] = adapted["category_hierarchy"].map(
        _extract_top_level_category
    ).astype("string")

    if "product_description" in adapted.columns:
        adapted["description"] = adapted[
            "product_description"
        ].astype("string")
    else:
        adapted["description"] = pd.Series(
            pd.NA,
            index=adapted.index,
            dtype="string",
        )

    if "product_features" in adapted.columns:
        adapted["attributes"] = adapted[
            "product_features"
        ].astype("string")
    else:
        adapted["attributes"] = pd.Series(
            pd.NA,
            index=adapted.index,
            dtype="string",
        )

    columns = [
        "product_id",
        "title",
        "description",
        "category",
        "subcategory",
        "attributes",
        "category_hierarchy",
    ]

    for column in [
        "rating_count",
        "average_rating",
        "review_count",
    ]:
        if column in adapted.columns:
            columns.append(column)

    return adapted[columns].reset_index(drop=True)


def load_wands_products(path: str | Path) -> pd.DataFrame:
    """Load and adapt the original WANDS product TSV file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"WANDS product file not found: {path}")

    products_df = pd.read_csv(
        path,
        sep="\t",
        dtype={"product_id": "string"},
    )

    products_df = products_df.rename(
        columns={"category hierarchy": "category_hierarchy"}
    )

    return adapt_wands_products(products_df)


def _extract_top_level_category(value: object) -> str | None:
    """Extract the first nonempty category in the hierarchy."""
    if pd.isna(value):
        return None

    parts = [
        part.strip()
        for part in str(value).split("/")
        if part.strip()
    ]

    return parts[0] if parts else None
