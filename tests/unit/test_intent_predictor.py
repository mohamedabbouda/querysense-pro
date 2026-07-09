from __future__ import annotations

import pandas as pd

from querysense.data.intent_dataset import prepare_intent_dataset, split_intent_dataset
from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor
from querysense.query_understanding.intent_predictor import HybridIntentPredictor
from querysense.training.train_intent import train_intent_classifier


def _build_products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Nike Air Zoom Pegasus",
                "description": "Black running shoes for men",
                "brand": "Nike",
                "category": "Fashion",
                "subcategory": "Shoes",
                "color": "Black",
                "size": "44",
                "gender": "men",
                "condition": "new",
                "price": 89.99,
                "currency": "EUR",
                "attributes": "{}",
            },
            {
                "product_id": "p002",
                "title": "Sony WH-1000XM4",
                "description": "Black wireless headphones",
                "brand": "Sony",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 249.99,
                "currency": "EUR",
                "attributes": "{}",
            },
        ]
    )


def _build_predictor() -> HybridIntentPredictor:
    training_df = pd.DataFrame(
        [
            {"normalized_query": "nike black shoes", "intent": "product_search"},
            {"normalized_query": "sony black headphones", "intent": "product_search"},
            {"normalized_query": "nike shoes under 100", "intent": "price_search"},
            {"normalized_query": "sony headphones under 300", "intent": "price_search"},
            {"normalized_query": "nike", "intent": "brand_search"},
            {"normalized_query": "sony", "intent": "brand_search"},
            {"normalized_query": "shoes", "intent": "category_search"},
            {"normalized_query": "headphones", "intent": "category_search"},
            {"normalized_query": "new nike shoes", "intent": "filtered_product_search"},
            {"normalized_query": "used sony headphones", "intent": "filtered_product_search"},
        ]
    )

    dataset = prepare_intent_dataset(training_df)
    split = split_intent_dataset(dataset, test_size=0.5, random_seed=42)
    trained_model = train_intent_classifier(split)

    extractor = RuleBasedEntityExtractor.from_products(_build_products_df())

    return HybridIntentPredictor(
        model=trained_model.pipeline,
        entity_extractor=extractor,
    )


def test_predict_price_search_with_rule() -> None:
    predictor = _build_predictor()

    prediction = predictor.predict("nike shoes under 120")

    assert prediction.intent == "price_search"
    assert prediction.source == "rule"


def test_predict_brand_search_with_rule() -> None:
    predictor = _build_predictor()

    prediction = predictor.predict("sony")

    assert prediction.intent == "brand_search"
    assert prediction.source == "rule"


def test_predict_category_search_with_rule() -> None:
    predictor = _build_predictor()

    prediction = predictor.predict("headphones")

    assert prediction.intent == "category_search"
    assert prediction.source == "rule"


def test_predict_filtered_product_search_with_rule() -> None:
    predictor = _build_predictor()

    prediction = predictor.predict("new nike shoes")

    assert prediction.intent == "filtered_product_search"
    assert prediction.source == "rule"


def test_predict_product_search_with_rule() -> None:
    predictor = _build_predictor()

    prediction = predictor.predict("sony headphones")

    assert prediction.intent == "product_search"
    assert prediction.source == "rule"

def test_predict_brand_color_product_type_as_product_search() -> None:
    predictor = _build_predictor()

    prediction = predictor.predict("sony black headphones")

    assert prediction.intent == "product_search"
    assert prediction.source == "rule"    