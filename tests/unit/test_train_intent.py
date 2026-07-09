from __future__ import annotations

import pandas as pd

from querysense.data.intent_dataset import prepare_intent_dataset, split_intent_dataset
from querysense.training.train_intent import train_intent_classifier


def test_train_intent_classifier_predicts_known_intents() -> None:
    df = pd.DataFrame(
        [
            {"normalized_query": "nike black shoes", "intent": "product_search"},
            {"normalized_query": "adidas white shoes", "intent": "product_search"},
            {"normalized_query": "apple iphone black", "intent": "product_search"},
            {"normalized_query": "nike shoes under 100", "intent": "price_search"},
            {"normalized_query": "adidas shoes under 80", "intent": "price_search"},
            {"normalized_query": "iphone under 500", "intent": "price_search"},
            {"normalized_query": "nike", "intent": "brand_search"},
            {"normalized_query": "adidas", "intent": "brand_search"},
            {"normalized_query": "apple", "intent": "brand_search"},
            {"normalized_query": "shoes", "intent": "category_search"},
            {"normalized_query": "smartphones", "intent": "category_search"},
            {"normalized_query": "laptops", "intent": "category_search"},
        ]
    )

    dataset = prepare_intent_dataset(df)
    split = split_intent_dataset(dataset, test_size=0.34, random_seed=42)

    trained_model = train_intent_classifier(split)

    predictions = trained_model.pipeline.predict(
        [
            "nike shoes",
            "nike shoes under 120",
            "apple",
            "laptops",
        ]
    )

    assert len(predictions) == 4
    assert set(predictions).issubset(set(trained_model.labels))