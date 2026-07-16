from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from querysense.retrieval.semantic_text import add_product_semantic_text_column
from querysense.utils.text import normalize_text_basic


class TextEmbeddingModel(Protocol):
    """Protocol for text embedding models."""

    def encode(
        self,
        sentences: list[str] | str,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """Encode text into embeddings."""


@dataclass
class SemanticProductIndex:
    """Semantic embedding index for product retrieval."""

    products_df: pd.DataFrame
    embedding_model: TextEmbeddingModel
    semantic_text_column: str = "semantic_text"

    def __post_init__(self) -> None:
        self.products_df = add_product_semantic_text_column(
            self.products_df,
            output_column=self.semantic_text_column,
        )

        semantic_texts = self.products_df[self.semantic_text_column].tolist()

        self.embeddings = self.embedding_model.encode(
            semantic_texts,
            normalize_embeddings=True,
        )

        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

    @classmethod
    def from_model_name(
        cls,
        products_df: pd.DataFrame,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> SemanticProductIndex:
        """Build a semantic index from a SentenceTransformer model name."""
        embedding_model = SentenceTransformer(model_name)

        return cls(
            products_df=products_df,
            embedding_model=embedding_model,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Search products using semantic similarity."""
        normalized_query = normalize_text_basic(query)

        query_embedding = self.embedding_model.encode(
            normalized_query,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        scores = self.embeddings @ query_embedding

        scored_df = self.products_df.copy()
        scored_df["semantic_score"] = scores

        return scored_df.sort_values(
            by="semantic_score",
            ascending=False,
        ).head(top_k).reset_index(drop=True)