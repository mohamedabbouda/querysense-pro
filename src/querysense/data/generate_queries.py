from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

INTENT_PRODUCT_SEARCH = "product_search"
INTENT_FILTERED_SEARCH = "filtered_product_search"
INTENT_BRAND_SEARCH = "brand_search"
INTENT_PRICE_SEARCH = "price_search"
INTENT_CATEGORY_SEARCH = "category_search"


@dataclass(frozen=True)
class QueryExample:
    """Synthetic query example generated from a product."""

    query: str
    intent: str
    target_product_id: str
    brand: str
    category: str
    subcategory: str
    color: str
    size: str
    gender: str
    condition: str
    max_price: float | None


def _clean_text(value: Any) -> str:
    """Convert a value into a clean lowercase string."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _product_type_from_subcategory(subcategory: str) -> str:
    """Convert subcategory into a simple product type phrase."""
    mapping = {
        "shoes": "shoes",
        "smartphones": "phone",
        "headphones": "headphones",
        "wearables": "smartwatch",
        "jeans": "jeans",
        "dresses": "dress",
        "laptops": "laptop",
        "furniture": "furniture",
        "cameras": "camera",
        "jackets": "jacket",
        "coats": "coat",
        "kitchen": "air fryer",
        "appliances": "vacuum cleaner",
        "accessories": "accessory",
    }
    return mapping.get(subcategory.lower(), subcategory.lower())


def generate_queries_for_product(product: pd.Series) -> list[QueryExample]:
    """Generate multiple realistic search queries for one product row."""
    product_id = _clean_text(product["product_id"])
    brand = _clean_text(product["brand"])
    category = _clean_text(product["category"])
    subcategory = _clean_text(product["subcategory"])
    color = _clean_text(product["color"])
    size = _clean_text(product["size"])
    gender = _clean_text(product["gender"])
    condition = _clean_text(product["condition"])
    price = float(product["price"])

    product_type = _product_type_from_subcategory(subcategory)

    examples = [
        QueryExample(
            query=f"{brand} {color} {product_type}",
            intent=INTENT_PRODUCT_SEARCH,
            target_product_id=product_id,
            brand=brand,
            category=category,
            subcategory=subcategory,
            color=color,
            size="",
            gender="",
            condition="",
            max_price=None,
        ),
        QueryExample(
            query=f"{brand} {product_type} {size}",
            intent=INTENT_FILTERED_SEARCH,
            target_product_id=product_id,
            brand=brand,
            category=category,
            subcategory=subcategory,
            color="",
            size=size,
            gender="",
            condition="",
            max_price=None,
        ),
        QueryExample(
            query=f"{color} {product_type} for {gender}",
            intent=INTENT_FILTERED_SEARCH,
            target_product_id=product_id,
            brand="",
            category=category,
            subcategory=subcategory,
            color=color,
            size="",
            gender=gender,
            condition="",
            max_price=None,
        ),
        QueryExample(
            query=f"{condition} {brand} {product_type}",
            intent=INTENT_FILTERED_SEARCH,
            target_product_id=product_id,
            brand=brand,
            category=category,
            subcategory=subcategory,
            color="",
            size="",
            gender="",
            condition=condition,
            max_price=None,
        ),
        QueryExample(
    query=f"{brand} {product_type} under {int(price + 50)}",
    intent=INTENT_PRICE_SEARCH,
    target_product_id=product_id,
    brand=brand,
    category=category,
    subcategory=subcategory,
    color="",
    size="",
    gender="",
    condition="",
    max_price=float(int(price + 50)),
        ),
        QueryExample(
            query=f"{subcategory}",
            intent=INTENT_CATEGORY_SEARCH,
            target_product_id=product_id,
            brand="",
            category=category,
            subcategory=subcategory,
            color="",
            size="",
            gender="",
            condition="",
            max_price=None,
        ),
        QueryExample(
            query=f"{brand}",
            intent=INTENT_BRAND_SEARCH,
            target_product_id=product_id,
            brand=brand,
            category="",
            subcategory="",
            color="",
            size="",
            gender="",
            condition="",
            max_price=None,
        ),
    ]

    return examples


def generate_synthetic_queries(
    products_df: pd.DataFrame,
    random_seed: int = 42,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Generate synthetic query examples from a product DataFrame."""
    examples: list[QueryExample] = []

    for _, product in products_df.iterrows():
        examples.extend(generate_queries_for_product(product))

    records = [example.__dict__ for example in examples]
    query_df = pd.DataFrame(records)

    if shuffle:
        query_df = query_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return query_df