from __future__ import annotations

import pandas as pd
import pytest

from querysense.data.intent_dataset import (
    prepare_intent_dataset,
    split_intent_dataset,
)


def test_prepare_intent_dataset() -> None:
    df = pd.DataFrame(
        [
            {
                "normalized_query": "nike black shoes",
                "intent": "product_search",
                "extra_column": "ignored",
            },
            {
                "normalized_query": "nike shoes under 100",
                "intent": "price_search",
                "extra_column": "ignored",
            },
        ]
    )

    dataset = prepare_intent_dataset(df)

    assert list(dataset.columns) == ["normalized_query", "intent"]
    assert dataset.shape == (2, 2)


def test_prepare_intent_dataset_missing_columns_raises_error() -> None:
    df = pd.DataFrame(
        [
            {
                "query": "nike shoes",
                "label": "product_search",
            }
        ]
    )

    with pytest.raises(ValueError, match="Missing required intent dataset columns"):
        prepare_intent_dataset(df)


def test_split_intent_dataset() -> None:
    df = pd.DataFrame(
        [
            {"normalized_query": f"product query {i}", "intent": "product_search"}
            for i in range(10)
        ]
        + [
            {"normalized_query": f"brand query {i}", "intent": "brand_search"}
            for i in range(10)
        ]
    )

    dataset = prepare_intent_dataset(df)
    split = split_intent_dataset(dataset, test_size=0.2, random_seed=42)

    assert len(split.train_texts) == 16
    assert len(split.test_texts) == 4
    assert len(split.train_labels) == 16
    assert len(split.test_labels) == 4