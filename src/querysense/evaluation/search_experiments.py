from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from querysense.evaluation.search_evaluator import evaluate_search_service
from querysense.evaluation.search_relevance import SearchRelevanceExample
from querysense.retrieval.search_service import (
    ProductSearchService,
    ProductSearchServiceConfig,
)


@dataclass(frozen=True)
class SearchExperimentConfig:
    """Configuration for one search experiment."""

    name: str
    search_service_config: ProductSearchServiceConfig


def run_search_experiments(
    experiment_configs: list[SearchExperimentConfig],
    relevance_examples: list[SearchRelevanceExample],
    k: int = 10,
) -> pd.DataFrame:
    """Run multiple search experiments and return a comparison dataframe."""
    rows = []

    for experiment_config in experiment_configs:
        search_service = ProductSearchService(
            experiment_config.search_service_config
        )

        evaluation = evaluate_search_service(
            search_service=search_service,
            relevance_examples=relevance_examples,
            k=k,
        )

        rows.append(
            {
                "experiment_name": experiment_config.name,
                "precision_at_k": evaluation.summary.precision_at_k,
                "recall_at_k": evaluation.summary.recall_at_k,
                "mean_reciprocal_rank": evaluation.summary.mean_reciprocal_rank,
                "ndcg_at_k": evaluation.summary.ndcg_at_k,
                "num_queries": evaluation.summary.num_queries,
                "k": evaluation.summary.k,
            }
        )

    return pd.DataFrame(rows)