
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

RELEVANCE_GRADES = {
    "Irrelevant": 0,
    "Partial": 1,
    "Exact": 2,
}

REQUIRED_QUERY_COLUMNS = {
    "query_id",
    "query",
    "query_class",
}

REQUIRED_LABEL_COLUMNS = {
    "id",
    "query_id",
    "product_id",
    "label",
}

DuplicatePolicy = Literal["error", "exclude_conflicts"]


class WandsRelevanceError(ValueError):
    """Raised when WANDS relevance data is invalid."""


def _require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(dataframe.columns)

    if missing:
        raise WandsRelevanceError(
            f"Missing {name} columns: {sorted(missing)}"
        )


def _normalize_id_column(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    dataframe[column] = (
        dataframe[column].astype("string").str.strip()
    )

    if (
        dataframe[column].isna().any()
        or dataframe[column].eq("").any()
    ):
        raise WandsRelevanceError(
            f"Missing or empty values in {column}"
        )


def _resolve_duplicate_judgments(
    labels: pd.DataFrame,
    duplicate_policy: DuplicatePolicy,
) -> pd.DataFrame:
    """Resolve duplicate query-product judgments.

    error:
        Reject any repeated query-product pair.

    exclude_conflicts:
        Keep one judgment when all labels agree.
        Remove the entire pair when labels disagree.
    """
    keys = ["query_id", "product_id"]

    if duplicate_policy not in ("error", "exclude_conflicts"):
        raise WandsRelevanceError(
            f"Unknown duplicate policy: {duplicate_policy}"
        )

    duplicated = labels.duplicated(
        subset=keys,
        keep=False,
    )

    if duplicate_policy == "error":
        if duplicated.any():
            raise WandsRelevanceError(
                "Duplicate query-product judgments detected"
            )

        return labels.copy()

    distinct_label_counts = labels.groupby(keys)["label"].transform(
        "nunique"
    )

    nonconflicting = labels.loc[
        distinct_label_counts == 1
    ].copy()

    resolved = nonconflicting.drop_duplicates(
        subset=keys,
        keep="first",
    )

    return resolved.reset_index(drop=True)


def adapt_wands_relevance(
    queries_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    products_df: pd.DataFrame | None = None,
    duplicate_policy: DuplicatePolicy = "error",
) -> pd.DataFrame:
    """Join queries to human judgments and preserve graded labels."""
    _require_columns(
        queries_df,
        REQUIRED_QUERY_COLUMNS,
        "query",
    )

    _require_columns(
        labels_df,
        REQUIRED_LABEL_COLUMNS,
        "label",
    )

    queries = queries_df.copy()
    labels = labels_df.copy()

    _normalize_id_column(queries, "query_id")
    _normalize_id_column(labels, "id")
    _normalize_id_column(labels, "query_id")
    _normalize_id_column(labels, "product_id")

    if queries["query_id"].duplicated().any():
        raise WandsRelevanceError(
            "Duplicate query IDs detected"
        )

    if labels["id"].duplicated().any():
        raise WandsRelevanceError(
            "Duplicate judgment IDs detected"
        )

    queries["query"] = (
        queries["query"].astype("string").str.strip()
    )

    if (
        queries["query"].isna().any()
        or queries["query"].eq("").any()
    ):
        raise WandsRelevanceError(
            "Queries must not be missing or empty"
        )

    if labels["label"].isna().any():
        raise WandsRelevanceError(
            "Missing relevance labels detected"
        )

    unexpected_labels = (
        set(labels["label"].unique())
        - set(RELEVANCE_GRADES)
    )

    if unexpected_labels:
        raise WandsRelevanceError(
            f"Unknown relevance labels: {sorted(unexpected_labels)}"
        )

    unknown_queries = (
        set(labels["query_id"])
        - set(queries["query_id"])
    )

    if unknown_queries:
        raise WandsRelevanceError(
            f"Unknown query IDs: {len(unknown_queries)}"
        )

    if products_df is not None:
        _require_columns(
            products_df,
            {"product_id"},
            "product",
        )

        products = products_df[["product_id"]].copy()

        _normalize_id_column(
            products,
            "product_id",
        )

        if products["product_id"].duplicated().any():
            raise WandsRelevanceError(
                "Duplicate product IDs detected"
            )

        unknown_products = (
            set(labels["product_id"])
            - set(products["product_id"])
        )

        if unknown_products:
            raise WandsRelevanceError(
                f"Unknown product IDs: {len(unknown_products)}"
            )

    labels = _resolve_duplicate_judgments(
        labels,
        duplicate_policy=duplicate_policy,
    )

    result = labels.merge(
        queries[
            [
                "query_id",
                "query",
                "query_class",
            ]
        ],
        on="query_id",
        how="left",
        validate="many_to_one",
    )

    result["relevance_grade"] = (
        result["label"]
        .map(RELEVANCE_GRADES)
        .astype("int64")
    )

    return result[
        [
            "id",
            "query_id",
            "query",
            "query_class",
            "product_id",
            "label",
            "relevance_grade",
        ]
    ].reset_index(drop=True)


def load_wands_relevance(
    query_path: str | Path,
    label_path: str | Path,
    products_df: pd.DataFrame | None = None,
    duplicate_policy: DuplicatePolicy = "error",
) -> pd.DataFrame:
    """Load original WANDS TSV files and adapt relevance judgments."""
    queries_df = pd.read_csv(
        query_path,
        sep="\t",
        dtype={
            "query_id": "string",
        },
    )

    labels_df = pd.read_csv(
        label_path,
        sep="\t",
        dtype={
            "id": "string",
            "query_id": "string",
            "product_id": "string",
        },
    )

    return adapt_wands_relevance(
        queries_df=queries_df,
        labels_df=labels_df,
        products_df=products_df,
        duplicate_policy=duplicate_policy,
    )
