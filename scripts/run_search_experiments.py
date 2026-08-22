from __future__ import annotations

from querysense.config import get_project_root
from querysense.evaluation.search_experiments import (
    SearchExperimentConfig,
    run_search_experiments,
)
from querysense.evaluation.search_relevance import load_search_relevance_csv
from querysense.retrieval.search_service import ProductSearchServiceConfig


def main() -> None:
    project_root = get_project_root()

    model_path = project_root / "models" / "intent_classifier.joblib"
    products_path = project_root / "data" / "processed" / "products.parquet"
    relevance_path = project_root / "data" / "annotations" / "search_relevance.csv"

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "search_experiments.csv"

    relevance_examples = load_search_relevance_csv(relevance_path)

    experiment_configs = [
    SearchExperimentConfig(
        name="structured_only",
        search_service_config=ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
            use_bm25_search=False,
            use_semantic_search=False,
            use_query_expansion=False,
        ),
    ),
    SearchExperimentConfig(
        name="structured_bm25_no_expansion",
        search_service_config=ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
            use_bm25_search=True,
            use_semantic_search=False,
            use_query_expansion=False,
        ),
    ),
    SearchExperimentConfig(
        name="structured_bm25_with_expansion",
        search_service_config=ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
            use_bm25_search=True,
            use_semantic_search=False,
            use_query_expansion=True,
        ),
    ),
    ]

    results = run_search_experiments(
        experiment_configs=experiment_configs,
        relevance_examples=relevance_examples,
        k=10,
    )

    results.to_csv(output_path, index=False)

    print("Search experiments complete")
    print(results.to_string(index=False))
    print(f"Saved experiment results to: {output_path}")


if __name__ == "__main__":
    main()