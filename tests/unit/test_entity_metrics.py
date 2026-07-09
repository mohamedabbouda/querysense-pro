from __future__ import annotations

import pandas as pd
import pytest

from querysense.evaluation.entity_metrics import evaluate_entity_extraction


def test_evaluate_entity_extraction_basic_accuracy() -> None:
    df = pd.DataFrame(
        [
            {
                "brand": "Nike",
                "extracted_brand": "nike",
                "color": "black",
                "extracted_color": "black",
            },
            {
                "brand": "Apple",
                "extracted_brand": "apple",
                "color": "white",
                "extracted_color": "black",
            },
            {
                "brand": None,
                "extracted_brand": None,
                "color": None,
                "extracted_color": None,
            },
        ]
    )

    result = evaluate_entity_extraction(df, entity_fields=["brand", "color"])

    brand_metric = next(metric for metric in result.field_metrics if metric.field_name == "brand")
    color_metric = next(metric for metric in result.field_metrics if metric.field_name == "color")

    assert brand_metric.evaluated_examples == 2
    assert brand_metric.correct_examples == 2
    assert brand_metric.accuracy == 1.0

    assert color_metric.evaluated_examples == 2
    assert color_metric.correct_examples == 1
    assert color_metric.accuracy == 0.5

    assert result.overall_accuracy == 0.75


def test_evaluate_entity_extraction_normalizes_punctuation() -> None:
    df = pd.DataFrame(
        [
            {
                "brand": "H&M",
                "extracted_brand": "h m",
            },
            {
                "brand": "Levi's",
                "extracted_brand": "levi s",
            },
        ]
    )

    result = evaluate_entity_extraction(df, entity_fields=["brand"])

    brand_metric = result.field_metrics[0]

    assert brand_metric.evaluated_examples == 2
    assert brand_metric.correct_examples == 2
    assert brand_metric.accuracy == 1.0


def test_evaluate_entity_extraction_missing_column_raises_error() -> None:
    df = pd.DataFrame(
        [
            {
                "brand": "nike",
            }
        ]
    )

    with pytest.raises(ValueError, match="Missing predicted column"):
        evaluate_entity_extraction(df, entity_fields=["brand"])

       