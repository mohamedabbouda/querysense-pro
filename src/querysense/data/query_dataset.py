from __future__ import annotations

import json

import pandas as pd

from querysense.query_understanding.normalizer import QueryNormalizer


def add_normalized_query_columns(queries_df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized query fields to a query dataset.

    Adds:
    - normalized_query
    - tokens
    - corrected_tokens
    - corrections
    """
    normalizer = QueryNormalizer()

    normalized_queries: list[str] = []
    tokens: list[str] = []
    corrected_tokens: list[str] = []
    corrections: list[str] = []

    for query in queries_df["query"]:
        result = normalizer.normalize(query)

        normalized_queries.append(result.normalized_query)
        tokens.append(json.dumps(result.tokens))
        corrected_tokens.append(json.dumps(result.corrected_tokens))
        corrections.append(json.dumps(result.corrections))

    enriched_df = queries_df.copy()
    enriched_df.insert(1, "normalized_query", normalized_queries)
    enriched_df.insert(2, "tokens", tokens)
    enriched_df.insert(3, "corrected_tokens", corrected_tokens)
    enriched_df.insert(4, "corrections", corrections)

    return enriched_df