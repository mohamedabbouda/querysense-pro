
"""Generate lexical Learning-to-Rank features for WANDS."""

from __future__ import annotations

import numpy as np
import pandas as pd

from querysense.retrieval.bm25_index import BM25ProductIndex
from querysense.utils.text import normalize_text_basic

REQUIRED_JUDGMENT_COLUMNS = {
    "query_id",
    "query",
    "product_id",
    "relevance_grade",
}

REQUIRED_PRODUCT_COLUMNS = {
    "product_id",
    "title",
}


class WandsFeatureError(ValueError):
    """Raised when LTR feature generation receives invalid data."""


def _tokens(text: str) -> list[str]:
    """Use the same normalization as our existing BM25 index."""
    return normalize_text_basic(text).split()


def _normalize_ids(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Normalize identifiers and reject missing values."""
    ids = dataframe[column].astype("string").str.strip()

    if ids.isna().any() or ids.eq("").any():
        raise WandsFeatureError(
            f"Missing or empty values in {column}"
        )

    return ids


def build_wands_ltr_features(
    judgments_df: pd.DataFrame,
    bm25_index: BM25ProductIndex,
) -> pd.DataFrame:
    """Generate BM25 and lexical features for judged pairs.

    BM25 corpus statistics come from the entire product index.
    Only human-judged query-product pairs receive feature rows.

    Generated features:
        bm25_score:
            BM25 relevance score for the query-product pair.

        title_token_overlap:
            Number of distinct query tokens appearing in the title.

        query_coverage:
            Fraction of distinct query tokens appearing in the
            product's searchable text.
    """
    missing = REQUIRED_JUDGMENT_COLUMNS - set(
        judgments_df.columns
    )

    if missing:
        raise WandsFeatureError(
            f"Missing judgment columns: {sorted(missing)}"
        )

    products = bm25_index.products_df

    missing_products = REQUIRED_PRODUCT_COLUMNS - set(
        products.columns
    )

    if missing_products:
        raise WandsFeatureError(
            f"Missing product columns: {sorted(missing_products)}"
        )

    if judgments_df.empty:
        raise WandsFeatureError(
            "No judgments provided"
        )

    # Reset the index so every row has a unique positional index.
    # This also handles inputs with duplicate DataFrame index labels.
    judgments = judgments_df.reset_index(drop=True).copy()

    judgments["query_id"] = _normalize_ids(
        judgments,
        "query_id",
    )

    judgments["product_id"] = _normalize_ids(
        judgments,
        "product_id",
    )

    judgments["query"] = (
        judgments["query"]
        .astype("string")
        .str.strip()
    )

    if (
        judgments["query"].isna().any()
        or judgments["query"].eq("").any()
    ):
        raise WandsFeatureError(
            "Missing or empty queries"
        )

    # Validate supervised relevance grades.
    if (
        judgments["relevance_grade"].isna().any()
        or not judgments["relevance_grade"]
        .isin([0, 1, 2])
        .all()
    ):
        raise WandsFeatureError(
            "Relevance grades must be 0, 1, or 2"
        )

    # Every query-product pair must have one judgment.
    if judgments.duplicated(
        ["query_id", "product_id"]
    ).any():
        raise WandsFeatureError(
            "Duplicate query-product pairs detected"
        )

    # Every query ID must correspond to one query text.
    if (
        judgments.groupby("query_id")["query"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise WandsFeatureError(
            "A query ID maps to multiple query texts"
        )

    # Normalize product IDs.
    product_ids = _normalize_ids(
        products,
        "product_id",
    )

    if product_ids.duplicated().any():
        raise WandsFeatureError(
            "Duplicate product IDs"
        )

    # Map WANDS product IDs to BM25 corpus row positions.
    # Product IDs must never be treated as array positions.
    product_positions = {
        product_id: position
        for position, product_id in enumerate(product_ids)
    }

    unknown_ids = (
        set(judgments["product_id"])
        - set(product_positions)
    )

    if unknown_ids:
        raise WandsFeatureError(
            f"Unknown product IDs: {len(unknown_ids)}"
        )

    # Precompute token sets for all indexed products.
    title_sets = [
        set(
            _tokens(
                "" if pd.isna(title) else str(title)
            )
        )
        for title in products["title"]
    ]

    search_sets = [
        set(_tokens(text))
        for text in products[
            bm25_index.search_text_column
        ]
    ]

    # Preserve judgment order and original columns.
    output = judgments.copy()

    # Initialize the generated numerical features.
    output["bm25_score"] = np.nan
    output["title_token_overlap"] = 0
    output["query_coverage"] = 0.0

    score_column = output.columns.get_loc(
        "bm25_score"
    )

    overlap_column = output.columns.get_loc(
        "title_token_overlap"
    )

    coverage_column = output.columns.get_loc(
        "query_coverage"
    )

    # Process one query group at a time.
    for _, group in judgments.groupby(
        "query_id",
        sort=False,
    ):
        query = str(group["query"].iloc[0])

        query_tokens = _tokens(query)
        query_set = set(query_tokens)

        # Find corpus positions for judged products.
        positions = [
            product_positions[product_id]
            for product_id in group["product_id"]
        ]

        # Score only judged products for this query.
        # BM25 statistics still use the full indexed catalog.
        scores = bm25_index.index.get_batch_scores(
            query_tokens,
            positions,
        )

        # Group indices now correspond to output row positions.
        output.iloc[
            group.index.to_numpy(),
            score_column,
        ] = scores

        # Calculate simple lexical features.
        for row_index, product_position in zip(
            group.index,
            positions,
            strict=True,
        ):
            output_position = int(row_index)

            title_overlap = len(
                query_set & title_sets[product_position]
            )

            if query_set:
                query_coverage = (
                    len(
                        query_set
                        & search_sets[product_position]
                    )
                    / len(query_set)
                )
            else:
                query_coverage = 0.0

            output.iat[
                output_position,
                overlap_column,
            ] = title_overlap

            output.iat[
                output_position,
                coverage_column,
            ] = query_coverage

    # Verify all BM25 scores were assigned successfully.
    bm25_scores = output[
        "bm25_score"
    ].to_numpy(dtype=float)

    if not np.isfinite(bm25_scores).all():
        raise WandsFeatureError(
            "Non-finite BM25 scores detected"
        )

    return output.reset_index(drop=True)
