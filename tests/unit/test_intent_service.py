from __future__ import annotations

from pathlib import Path

import pandas as pd

from querysense.data.intent_dataset import prepare_intent_dataset, split_intent_dataset
from querysense.query_understanding.intent_service import (
    IntentPredictionService,
    IntentServiceConfig,
)
from querysense.training.train_intent import save_intent_classifier, train_intent_classifier


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


def _train_test_model(tmp_path: Path) -> tuple[Path, Path]:
    products_path = tmp_path / "products.parquet"
    model_path = tmp_path / "intent_classifier.joblib"
    labels_path = tmp_path / "intent_labels.json"

    _build_products_df().to_parquet(products_path)

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

    save_intent_classifier(
        trained_model,
        model_output_path=model_path,
        label_output_path=labels_path,
    )

    return model_path, products_path


def test_intent_prediction_service_predicts_single_query(tmp_path: Path) -> None:
    model_path, products_path = _train_test_model(tmp_path)

    service = IntentPredictionService(
        IntentServiceConfig(
            model_path=model_path,
            products_path=products_path,
        )
    )

    prediction = service.predict("sony black headphones")

    assert prediction.intent == "product_search"
    assert prediction.normalized_query == "sony black headphones"


def test_intent_prediction_service_predicts_batch(tmp_path: Path) -> None:
    model_path, products_path = _train_test_model(tmp_path)

    service = IntentPredictionService(
        IntentServiceConfig(
            model_path=model_path,
            products_path=products_path,
        )
    )

    predictions = service.predict_batch(
        [
            "sony",
            "headphones",
            "sony headphones under 300",
        ]
    )

    assert [prediction.intent for prediction in predictions] == [
        "brand_search",
        "category_search",
        "price_search",
    ]