
from __future__ import annotations

import pandas as pd
import pytest

from querysense.data.wands_relevance import (
    WandsRelevanceError,
    adapt_wands_relevance,
    load_wands_relevance,
)


def make_queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "0",
                "query": "salon chair",
                "query_class": "Massage Chairs",
            },
            {
                "query_id": "1",
                "query": "office desk",
                "query_class": None,
            },
        ]
    )


def make_labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "0",
                "query_id": "0",
                "product_id": "100",
                "label": "Exact",
            },
            {
                "id": "1",
                "query_id": "0",
                "product_id": "101",
                "label": "Partial",
            },
            {
                "id": "2",
                "query_id": "1",
                "product_id": "102",
                "label": "Irrelevant",
            },
        ]
    )


def test_relevance_adapter_preserves_grades() -> None:
    result = adapt_wands_relevance(
        make_queries(),
        make_labels(),
    )

    assert result["relevance_grade"].tolist() == [2, 1, 0]

    assert result["query"].tolist() == [
        "salon chair",
        "salon chair",
        "office desk",
    ]

    assert pd.isna(result.iloc[2]["query_class"])


def test_relevance_adapter_rejects_unknown_labels() -> None:
    labels = make_labels()
    labels.loc[0, "label"] = "Unknown"

    with pytest.raises(
        WandsRelevanceError,
        match="Unknown relevance labels",
    ):
        adapt_wands_relevance(
            make_queries(),
            labels,
        )


def test_relevance_adapter_rejects_unknown_queries() -> None:
    labels = make_labels()
    labels.loc[0, "query_id"] = "999"

    with pytest.raises(
        WandsRelevanceError,
        match="Unknown query IDs",
    ):
        adapt_wands_relevance(
            make_queries(),
            labels,
        )


def test_relevance_adapter_rejects_duplicate_pairs() -> None:
    labels = make_labels()
    labels.loc[1, "product_id"] = "100"

    with pytest.raises(
        WandsRelevanceError,
        match="Duplicate query-product",
    ):
        adapt_wands_relevance(
            make_queries(),
            labels,
        )


def test_relevance_adapter_checks_product_ids() -> None:
    products = pd.DataFrame(
        {
            "product_id": ["100", "101"],
        }
    )

    with pytest.raises(
        WandsRelevanceError,
        match="Unknown product IDs",
    ):
        adapt_wands_relevance(
            make_queries(),
            make_labels(),
            products_df=products,
        )


def test_relevance_loader_reads_tsv(tmp_path) -> None:
    query_path = tmp_path / "query.csv"
    label_path = tmp_path / "label.csv"

    make_queries().to_csv(
        query_path,
        sep="\t",
        index=False,
    )

    make_labels().to_csv(
        label_path,
        sep="\t",
        index=False,
    )

    result = load_wands_relevance(
        query_path,
        label_path,
    )

    assert len(result) == 3
    assert result["relevance_grade"].tolist() == [2, 1, 0]


def test_relevance_adapter_deduplicates_agreeing_labels() -> None:
    labels = make_labels()

    duplicate = labels.iloc[[0]].copy()
    duplicate["id"] = "999"

    labels = pd.concat(
        [labels, duplicate],
        ignore_index=True,
    )

    result = adapt_wands_relevance(
        make_queries(),
        labels,
        duplicate_policy="exclude_conflicts",
    )

    assert len(result) == 3

    assert not result.duplicated(
        ["query_id", "product_id"]
    ).any()

    assert result["relevance_grade"].tolist() == [2, 1, 0]


def test_relevance_adapter_excludes_conflicting_pairs() -> None:
    labels = make_labels()

    duplicate = labels.iloc[[0]].copy()
    duplicate["id"] = "999"
    duplicate["label"] = "Partial"

    labels = pd.concat(
        [labels, duplicate],
        ignore_index=True,
    )

    result = adapt_wands_relevance(
        make_queries(),
        labels,
        duplicate_policy="exclude_conflicts",
    )

    assert len(result) == 2

    assert not (
        (result["query_id"] == "0")
        & (result["product_id"] == "100")
    ).any()


def test_relevance_adapter_rejects_duplicate_judgment_ids() -> None:
    labels = make_labels()

    duplicate = labels.iloc[[0]].copy()
    duplicate["product_id"] = "999"

    labels = pd.concat(
        [labels, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        WandsRelevanceError,
        match="Duplicate judgment IDs",
    ):
        adapt_wands_relevance(
            make_queries(),
            labels,
            duplicate_policy="exclude_conflicts",
        )


def test_relevance_adapter_rejects_invalid_policy() -> None:
    with pytest.raises(
        WandsRelevanceError,
        match="Unknown duplicate policy",
    ):
        adapt_wands_relevance(
            make_queries(),
            make_labels(),
            duplicate_policy="majority_vote",  # type: ignore[arg-type]
        )
