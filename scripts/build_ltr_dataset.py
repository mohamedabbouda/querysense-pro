from __future__ import annotations

from querysense.config import get_project_root
from querysense.evaluation.search_relevance import load_search_relevance_csv
from querysense.ranking.ltr_dataset import build_ltr_feature_dataset
from querysense.retrieval.search_service import (
    ProductSearchService,
    ProductSearchServiceConfig,
)


def main() -> None:
    project_root = get_project_root()

    model_path = project_root / "models" / "intent_classifier.joblib"
    products_path = project_root / "data" / "processed" / "products.parquet"
    relevance_path = project_root / "data" / "annotations" / "search_relevance.csv"
    output_path = project_root / "data" / "processed" / "ltr_feature_dataset.csv"

    relevance_examples = load_search_relevance_csv(relevance_path)

    search_service = ProductSearchService(
        ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
            max_results=10,
            use_bm25_search=True,
            use_semantic_search=False,
            use_query_expansion=True,
        )
    )

    dataset = build_ltr_feature_dataset(
        search_service=search_service,
        relevance_examples=relevance_examples,
    )

    dataset.to_csv(output_path, index=False)

    print("LTR feature dataset created")
    print(f"Rows: {len(dataset)}")
    print(f"Positive rows: {int(dataset['label'].sum())}")
    print(f"Negative rows: {int((dataset['label'] == 0).sum())}")
    print(f"Output: {output_path}")

    print("\nLabel distribution:")
    print(dataset["label"].value_counts().sort_index())

    print("\nFeature preview:")
    print(
        dataset[
            [
                "query",
                "product_id",
                "label",
                "manual_score",
                "bm25_score",
                "semantic_score",
                "brand_match",
                "subcategory_match",
                "title_token_overlap_count",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()