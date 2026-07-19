from __future__ import annotations

import json
from dataclasses import asdict

from querysense.config import get_project_root
from querysense.evaluation.search_evaluator import evaluate_search_service
from querysense.evaluation.search_relevance import load_search_relevance_csv
from querysense.retrieval.search_service import (
    ProductSearchService,
    ProductSearchServiceConfig,
)


def main() -> None:
    project_root = get_project_root()

    model_path = project_root / "models" / "intent_classifier.joblib"
    products_path = project_root / "data" / "processed" / "products.parquet"
    relevance_path = project_root / "data" / "annotations" / "search_relevance.csv"

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_dir / "search_metrics.json"
    per_query_path = reports_dir / "search_per_query_metrics.csv"

    relevance_examples = load_search_relevance_csv(relevance_path)

    search_service = ProductSearchService(
        ProductSearchServiceConfig(
            model_path=model_path,
            products_path=products_path,
            max_results=10,
        )
    )

    evaluation = evaluate_search_service(
        search_service=search_service,
        relevance_examples=relevance_examples,
        k=10,
    )

    summary = asdict(evaluation.summary)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    evaluation.per_query_results.to_csv(
        per_query_path,
        index=False,
    )

    print("Search evaluation complete")
    print(f"Queries evaluated: {evaluation.summary.num_queries}")
    print(f"Precision@{evaluation.summary.k}: {evaluation.summary.precision_at_k:.4f}")
    print(f"Recall@{evaluation.summary.k}: {evaluation.summary.recall_at_k:.4f}")
    print(f"MRR: {evaluation.summary.mean_reciprocal_rank:.4f}")
    print(f"NDCG@{evaluation.summary.k}: {evaluation.summary.ndcg_at_k:.4f}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved per-query metrics to: {per_query_path}")


if __name__ == "__main__":
    main()