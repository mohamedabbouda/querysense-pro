from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedEntities:
    """Structured entities extracted from a search query."""

    brand: str | None = None
    category: str | None = None
    subcategory: str | None = None
    product_type: str | None = None
    color: str | None = None
    size: str | None = None
    gender: str | None = None
    condition: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    price_intent: str | None = None

    def to_dict(self) -> dict[str, str | float | None]:
        """Convert extracted entities to a dictionary."""
        return {
            "brand": self.brand,
            "category": self.category,
            "subcategory": self.subcategory,
            "product_type": self.product_type,
            "color": self.color,
            "size": self.size,
            "gender": self.gender,
            "condition": self.condition,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "price_intent": self.price_intent,
        }

    def non_empty_entities(self) -> dict[str, str | float]:
        """Return only entities that were actually extracted."""
        return {
            key: value
            for key, value in self.to_dict().items()
            if value is not None and value != ""
        }