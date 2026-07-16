from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from querysense.retrieval.semantic_index import (
    SemanticProductIndex,
    TextEmbeddingModel,
)


@dataclass(frozen=True)
class SemanticSearchServiceConfig:
    """Configuration for semantic product search."""

    products_path: Path
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_results: int = 10


class SemanticSearchService:
    """Service for semantic product retrieval."""

    def __init__(
        self,
        config: SemanticSearchServiceConfig,
        embedding_model: TextEmbeddingModel | None = None,
    ) -> None:
        self.config = config
        self.products_df = pd.read_parquet(config.products_path)

        if embedding_model is None:
            self.semantic_index = SemanticProductIndex.from_model_name(
                products_df=self.products_df,
                model_name=config.model_name,
            )
        else:
            self.semantic_index = SemanticProductIndex(
                products_df=self.products_df,
                embedding_model=embedding_model,
            )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Search products using semantic retrieval."""
        max_results = top_k or self.config.max_results

        return self.semantic_index.search(
            query=query,
            top_k=max_results,
        )