from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ProductCondition = Literal["new", "used", "refurbished"]
Gender = Literal["men", "women", "unisex"]


class Product(BaseModel):
    """Validated product record used across QuerySense Pro."""

    product_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    brand: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    subcategory: str = Field(..., min_length=1)
    color: str = Field(..., min_length=1)
    size: str = Field(..., min_length=1)
    gender: Gender
    condition: ProductCondition
    price: float = Field(..., ge=0)
    currency: str = Field(..., min_length=3, max_length=3)
    attributes: str = Field(default="")

    @field_validator(
        "product_id",
        "title",
        "description",
        "brand",
        "category",
        "subcategory",
        "color",
        "size",
        "currency",
        mode="before",
    )
    @classmethod
    def strip_string_fields(cls, value: str) -> str:
        """Remove extra whitespace from string fields."""
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("condition", mode="before")
    @classmethod
    def normalize_condition(cls, value: str) -> str:
        """Normalize product condition."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, value: str) -> str:
        """Normalize gender field."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize currency code."""
        if isinstance(value, str):
            return value.strip().upper()
        return value


REQUIRED_PRODUCT_COLUMNS = [
    "product_id",
    "title",
    "description",
    "brand",
    "category",
    "subcategory",
    "color",
    "size",
    "gender",
    "condition",
    "price",
    "currency",
    "attributes",
]