from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SearchRelevanceExample:
    """A search query with known relevant products."""

    query: str
    relevant_product_ids: set[str]


def load_search_relevance_csv(path: Path) -> list[SearchRelevanceExample]:
    """Load search relevance labels from CSV."""
    relevance_df = pd.read_csv(path)

    _validate_relevance_columns(relevance_df)

    return [
        SearchRelevanceExample(
            query=str(row["query"]),
            relevant_product_ids=_parse_product_ids(row["relevant_product_ids"]),
        )
        for _, row in relevance_df.iterrows()
    ]


def _validate_relevance_columns(relevance_df: pd.DataFrame) -> None:
    required_columns = {"query", "relevant_product_ids"}
    missing_columns = required_columns - set(relevance_df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required relevance columns: {missing}")


def _parse_product_ids(value: object) -> set[str]:
    if pd.isna(value):
        return set()

    return {
        product_id.strip()
        for product_id in str(value).split(";")
        if product_id.strip()
    }