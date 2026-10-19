
from __future__ import annotations

import pandas as pd
import pytest

from querysense.ranking.ltr_split import split_ltr_dataset


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query": "nike running shoes", "product_id": "P001", "label": 1},
            {"query": "nike running shoes", "product_id": "P002", "label": 0},
            {"query": "running sneakers", "product_id": "P003", "label": 1},
            {"query": "iphone", "product_id": "P004", "label": 1},
            {"query": "iphone", "product_id": "P005", "label": 0},
            {"query": "office chair", "product_id": "P006", "label": 1},
            {"query": "camera", "product_id": "P007", "label": 1},
        ]
    )


def make_groups() -> dict[str, str]:
    return {
        "nike running shoes": "shoes",
        "running sneakers": "shoes",
        "iphone": "phones",
        "office chair": "furniture",
        "camera": "cameras",
    }


def test_split_preserves_all_rows() -> None:
    dataset = make_dataset()

    split = split_ltr_dataset(dataset, make_groups())

    total_rows = (
        len(split.train)
        + len(split.validation)
        + len(split.test)
    )

    assert total_rows == len(dataset)


def test_no_query_group_appears_in_multiple_splits() -> None:
    split = split_ltr_dataset(make_dataset(), make_groups())

    train_groups = set(split.train["query_group"])
    validation_groups = set(split.validation["query_group"])
    test_groups = set(split.test["query_group"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_related_queries_stay_together() -> None:
    split = split_ltr_dataset(make_dataset(), make_groups())

    partitions = [split.train, split.validation, split.test]

    for partition in partitions:
        queries = set(partition["query"])

        if "nike running shoes" in queries:
            assert "running sneakers" in queries


def test_split_is_reproducible() -> None:
    first = split_ltr_dataset(make_dataset(), make_groups(), seed=42)
    second = split_ltr_dataset(make_dataset(), make_groups(), seed=42)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_missing_group_assignment_raises_error() -> None:
    with pytest.raises(ValueError, match="Missing query-group"):
        split_ltr_dataset(make_dataset(), {"iphone": "phones"})


def test_too_few_groups_raises_error() -> None:
    dataset = make_dataset()

    groups = {
        query: "one_group"
        for query in dataset["query"].unique()
    }

    with pytest.raises(ValueError, match="At least three"):
        split_ltr_dataset(dataset, groups)
