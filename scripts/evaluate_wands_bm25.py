
"""Evaluate BM25 ranking quality on WANDS judged candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from querysense.evaluation.graded_ranking_metrics import graded_ndcg_at_k

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "wands" / "features"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "wands_bm25"

SPLIT_NAMES = ("train", "validation")
K = 10

REQUIRED_COLUMNS = {
    "query_id",
    "query",
    "product_id",
    "bm25_score",
    "relevance_grade",
}


def evaluate_split(
    features: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Calculate graded NDCG@K for each query."""

    missing = REQUIRED_COLUMNS - set(features.columns)

    if missing:
        raise ValueError(
            f"Missing feature columns: {sorted(missing)}"
        )

    if features.empty:
        raise ValueError("Feature dataset is empty")

    if (
        features["query_id"].isna().any()
        or features["product_id"].isna().any()
        or features["query"].isna().any()
    ):
        raise ValueError("Missing query or product identifiers")

    if features.duplicated(["query_id", "product_id"]).any():
        raise ValueError("Duplicate query-product pairs")

    if (
        features.groupby("query_id")["query"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Inconsistent query text for a query ID")

    if not features["relevance_grade"].isin([0, 1, 2]).all():
        raise ValueError("Invalid relevance grades")

    scores = features["bm25_score"].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("BM25 scores must be finite")

    rows = []

    for query_id, group in features.groupby("query_id", sort=True):
        # Product ID is a deterministic tie-breaker.
        ranked = group.sort_values(
            ["bm25_score", "product_id"],
            ascending=[False, True],
            kind="stable",
        )

        grades = ranked["relevance_grade"].astype(int).tolist()

        ndcg = graded_ndcg_at_k(grades, k=k)

        rows.append(
            {
                "query_id": query_id,
                "query": group["query"].iloc[0],
                "num_candidates": len(group),
                "num_exact": int(
                    (group["relevance_grade"] == 2).sum()
                ),
                "num_partial": int(
                    (group["relevance_grade"] == 1).sum()
                ),
                "num_irrelevant": int(
                    (group["relevance_grade"] == 0).sum()
                ),
                "ndcg_at_10": ndcg,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for split_name in SPLIT_NAMES:
        path = FEATURE_DIR / f"{split_name}_features.csv"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing feature dataset: {path}"
            )

        features = pd.read_csv(
            path,
            dtype={
                "query_id": "string",
                "product_id": "string",
            },
        )

        print(f"\nEvaluating {split_name}...")

        per_query = evaluate_split(features, k=K)

        mean_ndcg = float(per_query["ndcg_at_10"].mean())

        summary_rows.append(
            {
                "split": split_name,
                "queries": len(per_query),
                "judgments": len(features),
                "mean_ndcg_at_10": mean_ndcg,
            }
        )

        per_query.to_csv(
            OUTPUT_DIR / f"{split_name}_per_query.csv",
            index=False,
        )

        print(f"Queries: {len(per_query)}")
        print(f"Mean graded NDCG@{K}: {mean_ndcg:.4f}")

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "bm25_summary.csv",
        index=False,
    )

    print("\nBM25 baseline summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
