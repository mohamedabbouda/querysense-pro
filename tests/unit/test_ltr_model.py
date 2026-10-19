from __future__ import annotations

import pandas as pd
import pytest

from querysense.ranking.ltr_model import (
    LTR_FEATURE_COLUMNS,
    predict_ltr_scores,
    train_pointwise_ltr_model,
)


def _make_ltr_dataset() -> pd.DataFrame:
    rows = []

    positive_row = {
        "label": 1,
        "manual_score": 5.0,
        "bm25_score": 8.0,
        "semantic_score": 0.0,
        "price": 89.99,
        "brand_match": 1,
        "category_match": 1,
        "subcategory_match": 1,
        "color_match": 1,
        "size_match": 0,
        "gender_match": 0,
        "condition_match": 1,
        "has_bm25_match": 1,
        "has_semantic_match": 0,
        "title_token_overlap_count": 3,
        "title_token_overlap_ratio": 0.75,
    }

    negative_row = {
        "label": 0,
        "manual_score": 0.2,
        "bm25_score": 0.1,
        "semantic_score": 0.0,
        "price": 199.99,
        "brand_match": 0,
        "category_match": 0,
        "subcategory_match": 0,
        "color_match": 0,
        "size_match": 0,
        "gender_match": 0,
        "condition_match": 0,
        "has_bm25_match": 1,
        "has_semantic_match": 0,
        "title_token_overlap_count": 0,
        "title_token_overlap_ratio": 0.0,
    }

    rows.extend([positive_row, negative_row, positive_row, negative_row])

    return pd.DataFrame(rows)


def test_train_pointwise_ltr_model_trains_model() -> None:
    dataset = _make_ltr_dataset()

    result = train_pointwise_ltr_model(dataset)

    assert result.feature_columns == LTR_FEATURE_COLUMNS
    assert result.num_rows == 4
    assert result.num_positive_rows == 2
    assert result.num_negative_rows == 2
    assert hasattr(result.model, "predict_proba")


def test_predict_ltr_scores_returns_probabilities() -> None:
    dataset = _make_ltr_dataset()
    training_result = train_pointwise_ltr_model(dataset)

    scores = predict_ltr_scores(
        model=training_result.model,
        dataset=dataset,
    )

    assert len(scores) == len(dataset)
    assert scores.name == "ltr_score"
    assert scores.between(0.0, 1.0).all()


def test_train_pointwise_ltr_model_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        train_pointwise_ltr_model(pd.DataFrame())


def test_train_pointwise_ltr_model_requires_both_classes() -> None:
    dataset = _make_ltr_dataset()
    dataset["label"] = 1

    with pytest.raises(ValueError, match="both positive and negative"):
        train_pointwise_ltr_model(dataset)


def test_train_pointwise_ltr_model_requires_feature_columns() -> None:
    dataset = _make_ltr_dataset().drop(columns=["bm25_score"])

    with pytest.raises(ValueError, match="Missing LTR feature columns"):
        train_pointwise_ltr_model(dataset)