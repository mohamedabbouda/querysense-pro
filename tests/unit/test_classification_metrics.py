from __future__ import annotations

import pytest

from querysense.evaluation.classification_metrics import evaluate_classifier


def test_evaluate_classifier_perfect_predictions() -> None:
    true_labels = ["brand_search", "price_search", "product_search"]
    predicted_labels = ["brand_search", "price_search", "product_search"]

    result = evaluate_classifier(true_labels, predicted_labels)

    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.weighted_f1 == 1.0
    assert result.predictions["is_correct"].all()


def test_evaluate_classifier_partial_predictions() -> None:
    true_labels = ["brand_search", "price_search", "product_search"]
    predicted_labels = ["brand_search", "product_search", "product_search"]

    result = evaluate_classifier(true_labels, predicted_labels)

    assert result.accuracy == pytest.approx(2 / 3)
    assert result.predictions["is_correct"].tolist() == [True, False, True]


def test_evaluate_classifier_mismatched_lengths_raises_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_classifier(
            true_labels=["brand_search"],
            predicted_labels=["brand_search", "price_search"],
        )


def test_evaluate_classifier_empty_labels_raises_error() -> None:
    with pytest.raises(ValueError, match="empty label list"):
        evaluate_classifier(
            true_labels=[],
            predicted_labels=[],
        )