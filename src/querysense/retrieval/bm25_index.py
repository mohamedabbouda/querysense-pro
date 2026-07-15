from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rank_bm25 import BM25Okapi

from querysense.retrieval.product_text import add_product_search_text_column
from querysense.utils.text import normalize_text_basic


@dataclass
class BM25ProductIndex:
    """BM25 index for product keyword retrieval."""

    products_df: pd.DataFrame
    search_text_column: str = "search_text"

    def __post_init__(self) -> None:
        self.products_df = add_product_search_text_column(
            self.products_df,
            output_column=self.search_text_column,
        )
        self.tokenized_corpus = [
            _tokenize(text)
            for text in self.products_df[self.search_text_column].tolist()
        ]
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Search products using BM25 keyword ranking."""
        normalized_query = normalize_text_basic(query)
        query_tokens = _tokenize(normalized_query)

        scores = self.index.get_scores(query_tokens)

        scored_df = self.products_df.copy()
        scored_df["bm25_score"] = scores

        scored_df = scored_df[scored_df["bm25_score"] > 0]

        return scored_df.sort_values(
            by="bm25_score",
            ascending=False,
        ).head(top_k).reset_index(drop=True)


def _tokenize(text: str) -> list[str]:
    normalized_text = normalize_text_basic(text)
    return [
        token
        for token in normalized_text.split()
        if token
    ]