from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from querysense.retrieval.semantic_service import (
    SemanticSearchService,
    SemanticSearchServiceConfig,
)


class FakeEmbeddingModel:
    """Small deterministic embedding model for semantic service tests."""

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


def _write_products_parquet(tmp_path: Path) -> Path:
    products_path = tmp_path / "products.parquet"

    products_df = pd.DataFrame(
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

    products_df.to_parquet(products_path, index=False)

    return products_path


def test_semantic_search_service_returns_semantic_results(tmp_path: Path) -> None:
    products_path = _write_products_parquet(tmp_path)

    service = SemanticSearchService(
        config=SemanticSearchServiceConfig(
            products_path=products_path,
            max_results=2,
        ),
        embedding_model=FakeEmbeddingModel(),
    )

    result = service.search("noise blocking headset")

    assert len(result) == 2
    assert result.iloc[0]["product_id"] == "p001"
    assert "semantic_score" in result.columns


def test_semantic_search_service_respects_custom_top_k(tmp_path: Path) -> None:
    products_path = _write_products_parquet(tmp_path)

    service = SemanticSearchService(
        config=SemanticSearchServiceConfig(
            products_path=products_path,
            max_results=3,
        ),
        embedding_model=FakeEmbeddingModel(),
    )

    result = service.search("office furniture", top_k=1)

    assert len(result) == 1
    assert result.iloc[0]["product_id"] == "p003"