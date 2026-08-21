from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from querysense.data.intent_dataset import IntentDatasetSplit
from querysense.evaluation.search_experiments import (
    SearchExperimentConfig,
    run_search_experiments,
)
from querysense.evaluation.search_relevance import SearchRelevanceExample
from querysense.retrieval.search_service import ProductSearchServiceConfig
from querysense.training.train_intent import train_intent_classifier


def test_run_search_experiments_returns_comparison_dataframe(tmp_path: Path) -> None:
    model_path, products_path = _prepare_test_artifacts(tmp_path)

    experiment_configs = [
        SearchExperimentConfig(
            name="structured_only",
            search_service_config=ProductSearchServiceConfig(
                model_path=model_path,
                products_path=products_path,
                use_bm25_search=False,
                use_semantic_search=False,
            ),
        ),
        SearchExperimentConfig(
            name="structured_bm25",
            search_service_config=ProductSearchServiceConfig(
                model_path=model_path,
                products_path=products_path,
                use_bm25_search=True,
                use_semantic_search=False,
            ),
        ),
    ]

    relevance_examples = [
        SearchRelevanceExample(
            query="sony headphones",
            relevant_product_ids={"p001"},
        )
    ]

    result = run_search_experiments(
        experiment_configs=experiment_configs,
        relevance_examples=relevance_examples,
        k=2,
    )

    assert list(result["experiment_name"]) == [
        "structured_only",
        "structured_bm25",
    ]
    assert set(result.columns) == {
        "experiment_name",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
        "ndcg_at_k",
        "num_queries",
        "k",
    }
    assert result["num_queries"].tolist() == [1, 1]
    assert result["k"].tolist() == [2, 2]


def _prepare_test_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    products_path = tmp_path / "products.parquet"
    model_path = tmp_path / "intent_classifier.joblib"

    products_df = pd.DataFrame(
        [
            {
                "product_id": "p001",
                "title": "Sony Wireless Headphones",
                "description": "Black wireless headphones",
                "brand": "Sony",
                "category": "Electronics",
                "subcategory": "Headphones",
                "color": "Black",
                "size": "one-size",
                "gender": "unisex",
                "condition": "new",
                "price": 199.99,
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
        ]
    )

    split = IntentDatasetSplit(
        train_texts=[
            "sony headphones",
            "nike shoes",
            "cheap shoes",
            "electronics",
        ],
        test_texts=[
            "sony headphones",
        ],
        train_labels=[
            "product_search",
            "product_search",
            "price_search",
            "category_search",
        ],
        test_labels=[
            "product_search",
        ],
    )

    trained_model = train_intent_classifier(split)

    products_df.to_parquet(products_path, index=False)
    joblib.dump(trained_model.pipeline, model_path)

    return model_path, products_path