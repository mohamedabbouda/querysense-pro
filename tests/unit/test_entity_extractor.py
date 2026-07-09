from __future__ import annotations

import pandas as pd

from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor


def _load_extractor() -> RuleBasedEntityExtractor:
    products_df = pd.read_parquet("data/processed/products.parquet")
    return RuleBasedEntityExtractor.from_products(products_df)


def test_extract_brand_color_size_and_gender() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("nike black shoes men size 44")

    assert entities.brand == "nike"
    assert entities.color == "black"
    assert entities.size == "44"
    assert entities.gender == "men"


def test_extract_typo_corrected_entities() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("iphon blak")

    assert entities.brand == "apple"
    assert entities.color == "black"


def test_extract_multiword_brand() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("the north face black jacket size l")

    assert entities.brand == "the north face"
    assert entities.color == "black"
    assert entities.size == "l"


def test_extract_condition() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("used samsung phone")

    assert entities.brand == "samsung"
    assert entities.condition == "used"


def test_extract_subcategory_as_product_type() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("sony wireless headphones")

    assert entities.brand == "sony"
    assert entities.subcategory == "headphones"
    assert entities.product_type == "headphones"

def test_extract_brand_from_product_alias() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("galaxy green phone")

    assert entities.brand == "samsung"
    assert entities.color == "green"


def test_direct_brand_match_takes_priority_over_alias() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("apple magic keyboard white")

    assert entities.brand == "apple"
    assert entities.color == "white"    


def test_extract_max_price_under() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("nike shoes under 100")

    assert entities.brand == "nike"
    assert entities.max_price == 100.0


def test_extract_max_price_with_currency_symbol() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("laptop up to €700")

    assert entities.max_price == 700.0


def test_extract_min_price_over() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("camera over 300")

    assert entities.min_price == 300.0


def test_extract_price_intent_cheap() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("cheap headphones")

    assert entities.price_intent == "cheap"
    assert entities.subcategory == "headphones"


def test_extract_price_intent_expensive() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("premium laptop")

    assert entities.price_intent == "expensive"
    assert entities.subcategory == "laptops"

def test_extract_subcategory_from_product_type_alias() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("premium laptop")

    assert entities.price_intent == "expensive"
    assert entities.subcategory == "laptops"
    assert entities.product_type == "laptops"

def test_does_not_extract_size_from_hm_brand_fragment() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("new h&m coat")

    assert entities.brand == "h m"
    assert entities.subcategory == "coats"
    assert entities.size is None


def test_does_not_extract_size_from_levis_brand_fragment() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("levi's blue jeans")

    assert entities.brand == "levi s"
    assert entities.color == "blue"
    assert entities.subcategory == "jeans"
    assert entities.size is None


def test_extract_explicit_single_letter_size() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("zara red dress size m")

    assert entities.brand == "zara"
    assert entities.color == "red"
    assert entities.subcategory == "dresses"
    assert entities.size == "m"

def test_infers_category_from_subcategory() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("premium laptop")

    assert entities.subcategory == "laptops"
    assert entities.category == "electronics"


def test_infers_fashion_category_from_shoes() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("nike black shoes")

    assert entities.subcategory == "shoes"
    assert entities.category == "fashion"

def test_extract_accessory_alias() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("apple white accessory")

    assert entities.brand == "apple"
    assert entities.subcategory == "accessories"
    assert entities.category == "electronics"


def test_extract_clothing_size_without_explicit_size_word() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("zara dress m")

    assert entities.brand == "zara"
    assert entities.subcategory == "dresses"
    assert entities.size == "m"


def test_extract_jacket_size_without_explicit_size_word() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("the north face jacket l")

    assert entities.brand == "the north face"
    assert entities.subcategory == "jackets"
    assert entities.size == "l"


def test_still_avoids_hm_brand_fragment_as_size() -> None:
    extractor = _load_extractor()

    entities = extractor.extract("new h&m coat")

    assert entities.brand == "h m"
    assert entities.subcategory == "coats"
    assert entities.size is None
