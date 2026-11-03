
from __future__ import annotations

import pandas as pd
import pytest

from querysense.ranking.wands_split import (
    WandsSplitError,
    normalize_query_for_split,
    split_wands_relevance,
    summarize_wands_splits,
)


def make_relevance_data(n_queries: int = 20) -> pd.DataFrame:
    """Create synthetic query-product relevance judgments."""
    rows = []

    for query_id in range(n_queries):
        for product_id in range(3):
            rows.append(
                {
                    "query_id": str(query_id),
                    "query": f"search query {query_id}",
                    "product_id": str(product_id),
                    "relevance_grade": product_id,
                }
            )

    return pd.DataFrame(rows)


def test_normalize_query_for_split() -> None:
    assert normalize_query_for_split("  OFFICE   Chair ") == "office chair"


def test_split_preserves_all_rows() -> None:
    data = make_relevance_data()
    splits = split_wands_relevance(data)

    assert sum(len(part) for part in splits.values()) == len(data)


def test_split_prevents_query_leakage() -> None:
    splits = split_wands_relevance(make_relevance_data())

    query_sets = {
        name: set(part["query_id"])
        for name, part in splits.items()
    }

    assert query_sets["train"].isdisjoint(query_sets["validation"])
    assert query_sets["train"].isdisjoint(query_sets["test"])
    assert query_sets["validation"].isdisjoint(query_sets["test"])


def test_split_is_reproducible() -> None:
    data = make_relevance_data()

    first = split_wands_relevance(data, seed=42)
    second = split_wands_relevance(data, seed=42)

    for name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(first[name], second[name])


def test_split_keeps_normalized_duplicates_together() -> None:
    data = make_relevance_data()

    data.loc[data["query_id"] == "1", "query"] = "OFFICE Chair"
    data.loc[data["query_id"] == "2", "query"] = "office   chair"

    splits = split_wands_relevance(data)

    locations = {}

    for name, part in splits.items():
        for query_id in part["query_id"].unique():
            locations[query_id] = name

    assert locations["1"] == locations["2"]


def test_split_rejects_duplicate_judgments() -> None:
    data = make_relevance_data()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(
        WandsSplitError,
        match="Duplicate query-product",
    ):
        split_wands_relevance(data)


def test_split_rejects_invalid_ratios() -> None:
    with pytest.raises(
        WandsSplitError,
        match="sum to 1",
    ):
        split_wands_relevance(
            make_relevance_data(),
            train_ratio=0.8,
            validation_ratio=0.2,
            test_ratio=0.2,
        )


def test_split_rejects_inconsistent_query_text() -> None:
    data = make_relevance_data()
    data.loc[data.index[0], "query"] = "completely different"

    with pytest.raises(
        WandsSplitError,
        match="multiple normalized queries",
    ):
        split_wands_relevance(data)


def test_summarize_wands_splits() -> None:
    splits = split_wands_relevance(make_relevance_data())

    summary = summarize_wands_splits(splits)

    assert summary["judgments"].sum() == 60
    assert summary["queries"].sum() == 20
    assert summary["exact"].sum() == 20
