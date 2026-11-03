
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "query_id",
    "query",
    "product_id",
    "relevance_grade",
}

SPLIT_NAMES = ("train", "validation", "test")


class WandsSplitError(ValueError):
    """Raised when WANDS relevance data cannot be safely split."""


def normalize_query_for_split(query: str) -> str:
    """Normalize query text to detect equivalent query strings."""
    text = unicodedata.normalize("NFKC", query)
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def split_wands_relevance(
    relevance_df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split WANDS judgments into disjoint normalized-query groups.

    All judgments belonging to the same normalized query
    are assigned to the same partition.
    """
    missing = REQUIRED_COLUMNS - set(relevance_df.columns)

    if missing:
        raise WandsSplitError(
            f"Missing required columns: {sorted(missing)}"
        )

    ratios = [train_ratio, validation_ratio, test_ratio]

    if (
        any(not np.isfinite(ratio) or ratio <= 0 for ratio in ratios)
        or not np.isclose(sum(ratios), 1.0)
    ):
        raise WandsSplitError(
            "Split ratios must be positive and sum to 1"
        )

    if relevance_df.empty:
        raise WandsSplitError(
            "Cannot split an empty dataset"
        )

    data = relevance_df.copy()

    # Normalize identifiers to a consistent representation.
    for column in ("query_id", "product_id"):
        data[column] = (
            data[column].astype("string").str.strip()
        )

        if (
            data[column].isna().any()
            or data[column].eq("").any()
        ):
            raise WandsSplitError(
                f"Missing or empty values in {column}"
            )

    # Validate query text.
    data["query"] = (
        data["query"].astype("string").str.strip()
    )

    if (
        data["query"].isna().any()
        or data["query"].eq("").any()
    ):
        raise WandsSplitError(
            "Missing or empty queries"
        )

    # Validate supervised relevance grades.
    if data["relevance_grade"].isna().any():
        raise WandsSplitError(
            "Missing relevance grades"
        )

    if not data["relevance_grade"].isin([0, 1, 2]).all():
        raise WandsSplitError(
            "Relevance grades must be 0, 1, or 2"
        )

    # Each query-product pair must have one resolved judgment.
    if data.duplicated(["query_id", "product_id"]).any():
        raise WandsSplitError(
            "Duplicate query-product judgments detected"
        )

    # Compute a stable grouping key from query text.
    data["_normalized_query"] = (
        data["query"].map(normalize_query_for_split)
    )

    # Each query ID must correspond to only one normalized text.
    query_text_counts = (
        data.groupby("query_id")["_normalized_query"].nunique()
    )

    if (query_text_counts > 1).any():
        raise WandsSplitError(
            "A query ID maps to multiple normalized queries"
        )

    groups = data["_normalized_query"].unique()

    if len(groups) < 3:
        raise WandsSplitError(
            "At least three distinct query groups are required"
        )

    # Shuffle GROUPS, not individual judgment rows.
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(groups)

    n_groups = len(shuffled_groups)

    n_train = int(round(n_groups * train_ratio))
    n_validation = int(round(n_groups * validation_ratio))
    n_test = n_groups - n_train - n_validation

    if min(n_train, n_validation, n_test) < 1:
        raise WandsSplitError(
            "Each split must contain at least one query group"
        )

    train_groups = set(
        shuffled_groups[:n_train]
    )

    validation_groups = set(
        shuffled_groups[n_train:n_train + n_validation]
    )

    test_groups = set(
        shuffled_groups[n_train + n_validation:]
    )

    assigned_groups = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }

    splits = {}

    for name in SPLIT_NAMES:
        partition = data.loc[
            data["_normalized_query"].isin(
                assigned_groups[name]
            )
        ].copy()

        splits[name] = (
            partition.drop(columns=["_normalized_query"])
            .reset_index(drop=True)
        )

    return splits


def summarize_wands_splits(
    splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize the size and relevance labels of each split."""
    rows = []

    for name in SPLIT_NAMES:
        if name not in splits:
            raise WandsSplitError(
                f"Missing split: {name}"
            )

        data = splits[name]

        rows.append(
            {
                "split": name,
                "queries": data["query_id"].nunique(),
                "products": data["product_id"].nunique(),
                "judgments": len(data),
                "irrelevant": int(
                    (data["relevance_grade"] == 0).sum()
                ),
                "partial": int(
                    (data["relevance_grade"] == 1).sum()
                ),
                "exact": int(
                    (data["relevance_grade"] == 2).sum()
                ),
            }
        )

    return pd.DataFrame(rows)

