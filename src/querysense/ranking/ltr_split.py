
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


QUERY_GROUPS = {
    "sony black headphones": "headphones",
    "wireless noise cancelling": "headphones",
    "noise blocking headset": "headphones",
    "nike black running shoes": "running_shoes",
    "comfortable running shoes": "running_shoes",
    "something for jogging": "running_shoes",
    "iphone black": "smartphones",
    "phone for daily use": "smartphones",
    "apple smartphone": "smartphones",
    "office work laptop": "laptops",
    "business laptop": "laptops",
    "premium laptop": "laptops",
    "white shoes for women": "womens_shoes",
    "women white sneakers": "womens_shoes",
    "red summer dress": "dresses",
    "home office desk": "office_furniture",
    "office furniture": "office_furniture",
    "comfortable office chair": "office_furniture",
    "action camera": "cameras",
    "waterproof action camera": "cameras",
}


@dataclass(frozen=True)
class LTRDatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_ltr_dataset(
    dataset: pd.DataFrame,
    query_groups: dict[str, str] | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> LTRDatasetSplit:
    """Split by query families, not individual query-product rows."""

    if "query" not in dataset.columns:
        raise ValueError("Dataset must contain a query column")

    if dataset.empty:
        raise ValueError("Dataset must not be empty")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    if train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave room for testing")

    groups = QUERY_GROUPS if query_groups is None else query_groups

    queries = set(dataset["query"].astype(str))
    missing_queries = queries - set(groups)

    if missing_queries:
        raise ValueError(
            f"Missing query-group assignments: {sorted(missing_queries)}"
        )

    grouped = dataset.copy()
    grouped["query_group"] = grouped["query"].map(groups)

    group_names = (
        pd.Series(sorted(grouped["query_group"].unique()))
        .sample(frac=1, random_state=seed)
        .tolist()
    )

    if len(group_names) < 3:
        raise ValueError("At least three query groups are required")

    num_groups = len(group_names)

    train_count = max(1, int(num_groups * train_fraction))
    validation_count = max(1, int(num_groups * validation_fraction))

    if train_count + validation_count >= num_groups:
        raise ValueError("Not enough groups for all three splits")

    train_groups = set(group_names[:train_count])
    validation_groups = set(
        group_names[train_count : train_count + validation_count]
    )
    test_groups = set(
        group_names[train_count + validation_count :]
    )

    return LTRDatasetSplit(
        train=grouped[
            grouped["query_group"].isin(train_groups)
        ].reset_index(drop=True),
        validation=grouped[
            grouped["query_group"].isin(validation_groups)
        ].reset_index(drop=True),
        test=grouped[
            grouped["query_group"].isin(test_groups)
        ].reset_index(drop=True),
    )
