from __future__ import annotations

import numpy as np
import pandas as pd

from querysense.retrieval.semantic_index import SemanticProductIndex


class FakeEmbeddingModel:
    """Small deterministic embedding model for tests."""

    def encode(
        self,
        sentences: list[str] | str,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            return self._encode_one(sentences, normalize_embeddings)

        return np.asarray(
            [
                self._encode_one(sentence, normalize_embeddings)
                for sentence in sentences
            ],
            dtype=np.float32,
        )

    def _encode_one(
        self,
        sentence: str,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        sentence = sentence.lower()

        vector = np.asarray(
            [
                1.0 if "headphone" in sentence or "noise" in sentence else 0.0,
                1.0 if "shoe" in sentence or "running" in sentence else 0.0,
                1.0 if "desk" in sentence or "office" in sentence else 0.0,
            ],
            dtype=np.float32,
        )

        if normalize_embeddings:
            norm = np.linalg.norm(vector)

            if norm > 0:
                vector = vector / norm

        return vector


def _build_products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Sony WH-1000XM4 Wireless Headphones",
                "description": "Black wireless noise cancelling headphones",
                "brand": "Sony",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 249.99,
                "currency": "EUR",
            },
            {
                "product_id": "p002",
                "title": "Nike Running Shoes",
                "description": "Comfortable shoes for running",
                "brand": "Nike",
                "category": "Fashion",
                "subcategory": "Shoes",
                "color": "Black",
                "size": "44",
                "gender": "men",
                "condition": "new",
                "price": 89.99,
                "currency": "EUR",
            },
            {
                "product_id": "p003",
                "title": "IKEA Office Desk",
                "description": "White desk for home office",
                "brand": "IKEA",
                "category": "Home",
                "subcategory": "Furniture",
                "color": "White",
                "size": "120x60",
                "gender": "unisex",
                "condition": "new",
                "price": 129.99,
                "currency": "EUR",
            },
        ]
    )


def test_semantic_product_index_adds_semantic_text_column() -> None:
    index = SemanticProductIndex(
        products_df=_build_products_df(),
        embedding_model=FakeEmbeddingModel(),
    )

    assert "semantic_text" in index.products_df.columns
    assert index.embeddings.shape == (3, 3)


def test_semantic_product_index_returns_semantically_similar_product() -> None:
    index = SemanticProductIndex(
        products_df=_build_products_df(),
        embedding_model=FakeEmbeddingModel(),
    )

    result = index.search("noise blocking headset", top_k=2)

    assert result.iloc[0]["product_id"] == "p001"
    assert result.iloc[0]["semantic_score"] > result.iloc[1]["semantic_score"]


def test_semantic_product_index_respects_top_k() -> None:
    index = SemanticProductIndex(
        products_df=_build_products_df(),
        embedding_model=FakeEmbeddingModel(),
    )

    result = index.search("office work furniture", top_k=1)

    assert len(result) == 1
    assert result.iloc[0]["product_id"] == "p003"