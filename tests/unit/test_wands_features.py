
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from querysense.ranking.wands_features import (
    WandsFeatureError,
    build_wands_ltr_features,
)
from querysense.retrieval.bm25_index import BM25ProductIndex


def make_products() -> pd.DataFrame:
    """Create a small synthetic product catalog."""
    return pd.DataFrame(
        {
            "product_id": [
                "p30",
                "p10",
                "p50",
                "p20",
                "p40",
            ],
            "title": [
                "office chair",
                "office desk",
                "gaming chair",
                "dining table",
                "garden lamp",
            ],
            "description": [
                "comfortable office seating",
                "wooden furniture",
                "ergonomic gaming seat",
                "wooden dining furniture",
                "outdoor lighting",
            ],
        }
    )


def make_judgments() -> pd.DataFrame:
    """Create synthetic relevance judgments."""
    return pd.DataFrame(
        {
            "query_id": [
                "q1",
                "q1",
                "q1",
                "q2",
            ],
            "query": [
                "office chair",
                "office chair",
                "office chair",
                "garden lamp",
            ],
            "product_id": [
                "p10",
                "p30",
                "p40",
                "p40",
            ],
            "relevance_grade": [
                0,
                2,
                0,
                2,
            ],
        }
    )


def test_features_preserve_judgments() -> None:
    index = BM25ProductIndex(
        make_products()
    )

    result = build_wands_ltr_features(
        make_judgments(),
        index,
    )

    assert len(result) == 4

    assert result["product_id"].tolist() == [
        "p10",
        "p30",
        "p40",
        "p40",
    ]

    assert result["relevance_grade"].tolist() == [
        0,
        2,
        0,
        2,
    ]


def test_bm25_scores_match_full_corpus_scoring() -> None:
    index = BM25ProductIndex(
        make_products()
    )

    result = build_wands_ltr_features(
        make_judgments(),
        index,
    )

    positions = {
        product_id: position
        for position, product_id in enumerate(
            index.products_df["product_id"]
        )
    }

    for _, row in result.iterrows():
        expected = index.score_query(
            row["query"]
        )[positions[row["product_id"]]]

        assert row["bm25_score"] == pytest.approx(
            expected
        )


def test_title_overlap_and_query_coverage() -> None:
    index = BM25ProductIndex(
        make_products()
    )

    result = build_wands_ltr_features(
        make_judgments(),
        index,
    )

    office_chair = result.iloc[1]

    assert office_chair["title_token_overlap"] == 2

    assert office_chair["query_coverage"] == pytest.approx(
        1.0
    )

    garden_lamp = result.iloc[2]

    assert garden_lamp["title_token_overlap"] == 0

    assert garden_lamp["query_coverage"] == pytest.approx(
        0.0
    )


def test_unknown_product_is_rejected() -> None:
    index = BM25ProductIndex(
        make_products()
    )

    judgments = make_judgments()

    judgments.loc[0, "product_id"] = "missing"

    with pytest.raises(
        WandsFeatureError,
        match="Unknown product IDs",
    ):
        build_wands_ltr_features(
            judgments,
            index,
        )


def test_duplicate_judgments_are_rejected() -> None:
    index = BM25ProductIndex(
        make_products()
    )

    judgments = make_judgments()

    judgments = pd.concat(
        [
            judgments,
            judgments.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        WandsFeatureError,
        match="Duplicate query-product",
    ):
        build_wands_ltr_features(
            judgments,
            index,
        )


def test_generated_scores_are_finite() -> None:
    index = BM25ProductIndex(
        make_products()
    )

    result = build_wands_ltr_features(
        make_judgments(),
        index,
    )

    assert np.isfinite(
        result["bm25_score"]
    ).all()

    assert result["query_coverage"].between(
        0,
        1,
    ).all()


def test_features_support_duplicate_dataframe_indices() -> None:
    """Feature assignments must not depend on index labels."""
    index = BM25ProductIndex(
        make_products()
    )

    judgments = make_judgments()

    # Deliberately introduce duplicate DataFrame index labels.
    judgments.index = [0, 0, 1, 1]

    result = build_wands_ltr_features(
        judgments,
        index,
    )

    assert len(result) == 4

    assert result["product_id"].tolist() == [
        "p10",
        "p30",
        "p40",
        "p40",
    ]

    assert result["relevance_grade"].tolist() == [
        0,
        2,
        0,
        2,
    ]

    positions = {
        product_id: position
        for position, product_id in enumerate(
            index.products_df["product_id"]
        )
    }

    # Verify every BM25 score belongs to the correct product.
    for _, row in result.iterrows():
        expected_score = index.score_query(
            row["query"]
        )[positions[row["product_id"]]]

        assert row["bm25_score"] == pytest.approx(
            expected_score
        )
