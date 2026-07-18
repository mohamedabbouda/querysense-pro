from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from querysense.data.intent_dataset import prepare_intent_dataset, split_intent_dataset
from querysense.retrieval.search_service import ProductSearchService, ProductSearchServiceConfig
from querysense.training.train_intent import save_intent_classifier, train_intent_classifier


class FakeEmbeddingModel:
    """Small deterministic embedding model for search service tests."""

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
                "title": "Nike Air Zoom Pegasus",
                "description": "Black running shoes for men",
                "brand": "Nike",
                "category": "Fashion",
                "subcategory": "Shoes",
                "color": "Black",
                "size": "44",
                "gender": "men",
                "condition": "new",
                "price": 89.99,
                "currency": "EUR",
                "attributes": "{}",
            },
            {
                "product_id": "p002",
                "title": "Sony WH-1000XM4 Wireless Headphones",
                "description": "Black wireless headphones",
                "brand": "Sony",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 249.99,
                "currency": "EUR",
                "attributes": "{}",
            },
            {
                "product_id": "p003",
                "title": "Dell XPS 13",
                "description": "Silver laptop",
                "brand": "Dell",
                "category": "Electronics",
                "subcategory": "Laptops",
                "color": "Silver",
                "size": "13-inch",
                "gender": "unisex",
                "condition": "new",
                "price": 999.99,
                "currency": "EUR",
                "attributes": "{}",
            },
        ]
    )


def _train_test_model(tmp_path: Path) -> tuple[Path, Path]:
    products_path = tmp_path / "products.parquet"
    model_path = tmp_path / "intent_classifier.joblib"
    labels_path = tmp_path / "intent_labels.json"

    _build_products_df().to_parquet(products_path)

    training_df = pd.DataFrame(
        [
            {"normalized_query": "nike black shoes", "intent": "product_search"},
            {"normalized_query": "sony black headphones", "intent": "product_search"},
            {"normalized_query": "dell silver laptop", "intent": "product_search"},
            {"normalized_query": "nike shoes under 100", "intent": "price_search"},
            {"normalized_query": "sony headphones under 300", "intent": "price_search"},
            {"normalized_query": "dell laptop under 1200", "intent": "price_search"},
            {"normalized_query": "nike", "intent": "brand_search"},
            {"normalized_query": "sony", "intent": "brand_search"},
            {"normalized_query": "dell", "intent": "brand_search"},
            {"normalized_query": "shoes", "intent": "category_search"},
            {"normalized_query": "headphones", "intent": "category_search"},
            {"normalized_query": "laptops", "intent": "category_search"},
            {"normalized_query": "new nike shoes", "intent": "filtered_product_search"},
            {"normalized_query": "new sony headphones", "intent": "filtered_product_search"},
            {"normalized_query": "new dell laptop", "intent": "filtered_product_search"},
        ]
    )

    dataset = prepare_intent_dataset(training_df)
    split = split_intent_dataset(dataset, test_size=0.4, random_seed=42)
    trained_model = train_intent_classifier(split)

    save_intent_classifier(
        trained_model,
        model_output_path=model_path,
        label_output_path=labels_path,
    )

    return model_path, products_path


def test_product_search_service_returns_matching_result(tmp_path: Path) -> None:
    model_path, products_path = _train_test_model(tmp_path)

    service = ProductSearchService(
        ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
        )
    )

    response = service.search("sony black headphones under 300")

    assert response.query == "sony black headphones under 300"
    assert response.intent == "price_search"
    assert response.entities.brand == "sony"
    assert response.entities.subcategory == "headphones"
    assert response.entities.color == "black"
    assert response.entities.max_price == 300.0
    filters = {
        filter_.name: filter_.value
        for filter_ in response.recommended_filters
    }
    assert filters["brand"] == "sony"
    assert filters["subcategory"] == "headphones"
    assert filters["color"] == "black"
    assert filters["max_price"] == 300.0
    
    assert len(response.results) >= 1
    assert response.results[0].product_id == "p002"
    assert response.results[0].score > 0
    assert "brand" in response.results[0].match_reasons
    assert "subcategory" in response.results[0].match_reasons
    assert "max_price" in response.results[0].match_reasons


def test_product_search_service_uses_bm25_when_filters_are_too_strict(tmp_path: Path) -> None:
    model_path, products_path = _train_test_model(tmp_path)

    service = ProductSearchService(
        ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
        )
    )

    response = service.search("sony headphones under 100")

    assert response.intent == "price_search"
    assert len(response.results) >= 1
    assert response.results[0].product_id == "p002"

def test_product_search_service_respects_max_results(tmp_path: Path) -> None:
    model_path, products_path = _train_test_model(tmp_path)

    service = ProductSearchService(
        ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
            max_results=1,
        )
    )

    response = service.search("electronics")

    assert len(response.results) <= 1



def test_product_search_service_uses_bm25_for_keyword_query(tmp_path: Path) -> None:
    model_path, products_path = _train_test_model(tmp_path)

    service = ProductSearchService(
        ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
        )
    )

    response = service.search("wireless headphones")

    assert len(response.results) >= 1
    assert response.results[0].product_id == "p002"
    assert response.results[0].bm25_score > 0
    assert "bm25" in response.results[0].match_reasons


def test_product_search_service_can_use_semantic_candidates(tmp_path: Path) -> None:
    model_path, products_path = _train_test_model(tmp_path)

    service = ProductSearchService(
        ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
            use_semantic_search=True,
        ),
        embedding_model=FakeEmbeddingModel(),
    )

    response = service.search("noise blocking headset")

    semantic_matches = [
        result
        for result in response.results
        if result.semantic_score > 0
    ]

    assert semantic_matches
    assert semantic_matches[0].product_id == "p002"
    assert "semantic" in semantic_matches[0].match_reasons