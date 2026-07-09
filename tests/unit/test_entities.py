from __future__ import annotations

from querysense.query_understanding.entities import ExtractedEntities


def test_extracted_entities_defaults_to_none() -> None:
    entities = ExtractedEntities()

    assert entities.brand is None
    assert entities.color is None
    assert entities.size is None
    assert entities.max_price is None


def test_extracted_entities_to_dict() -> None:
    entities = ExtractedEntities(
        brand="nike",
        color="black",
        size="44",
        gender="men",
        max_price=100.0,
    )

    assert entities.to_dict() == {
        "brand": "nike",
        "category": None,
        "subcategory": None,
        "product_type": None,
        "color": "black",
        "size": "44",
        "gender": "men",
        "condition": None,
        "min_price": None,
        "max_price": 100.0,
        "price_intent": None,
    }


def test_extracted_entities_non_empty_entities() -> None:
    entities = ExtractedEntities(
        brand="apple",
        product_type="phone",
        color="black",
    )

    assert entities.non_empty_entities() == {
        "brand": "apple",
        "product_type": "phone",
        "color": "black",
    }