
"""Generate WANDS Learning-to-Rank feature datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from querysense.data.wands_adapter import load_wands_products
from querysense.ranking.wands_features import build_wands_ltr_features
from querysense.retrieval.bm25_index import BM25ProductIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WANDS_DIR = PROJECT_ROOT.parent / "WANDS" / "dataset"

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "wands"

OUTPUT_DIR = SPLIT_DIR / "features"

SPLIT_NAMES = ("train", "validation", "test")

FEATURE_COLUMNS = [
    "bm25_score",
    "title_token_overlap",
    "query_coverage",
]

EXPECTED_ROWS = 231_859
EXPECTED_QUERIES = 480


def load_judgments(split_name: str) -> pd.DataFrame:
    """Load one previously generated WANDS split."""
    path = SPLIT_DIR / f"{split_name}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing split file: {path}. "
            "Run scripts/split_wands.py first."
        )

    return pd.read_csv(
        path,
        dtype={
            "id": "string",
            "query_id": "string",
            "product_id": "string",
        },
    )


def validate_feature_dataset(
    original: pd.DataFrame,
    features: pd.DataFrame,
    split_name: str,
) -> None:
    """Ensure features preserve all judgments and are finite."""

    if len(original) != len(features):
        raise ValueError(
            f"{split_name}: Row count mismatch"
        )

    # The extractor must preserve the original query-product pairs.
    for column in ("query_id", "product_id"):
        expected = original[column].astype("string").reset_index(
            drop=True
        )
        actual = features[column].astype("string").reset_index(
            drop=True
        )

        if not expected.equals(actual):
            raise ValueError(
                f"{split_name}: {column} ordering changed"
            )

    # The human relevance labels must not change.
    if not np.array_equal(
        original["relevance_grade"].to_numpy(),
        features["relevance_grade"].to_numpy(),
    ):
        raise ValueError(
            f"{split_name}: Relevance grades changed"
        )

    if features.duplicated(
        ["query_id", "product_id"]
    ).any():
        raise ValueError(
            f"{split_name}: Duplicate query-product pairs"
        )

    # All numerical features must be finite.
    feature_values = features[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            f"{split_name}: Non-finite feature values"
        )

    if (features["title_token_overlap"] < 0).any():
        raise ValueError(
            f"{split_name}: Invalid title overlap"
        )

    if not features["query_coverage"].between(0, 1).all():
        raise ValueError(
            f"{split_name}: Invalid query coverage"
        )

    print(
        f"Validated {split_name}: "
        f"{len(features):,} rows, "
        f"{features['query_id'].nunique()} queries"
    )


def main() -> None:
    print("Loading WANDS product catalog...")

    products = load_wands_products(
        WANDS_DIR / "product.csv"
    )

    print(f"Products loaded: {len(products):,}")

    print("\nBuilding full-catalog BM25 index...")

    bm25_index = BM25ProductIndex(products)

    print(
        "Indexed products:",
        len(bm25_index.products_df),
    )

    feature_datasets: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for split_name in SPLIT_NAMES:
        print(f"\nProcessing {split_name}...")

        judgments = load_judgments(split_name)

        print(
            "Judgments:",
            f"{len(judgments):,}",
        )

        features = build_wands_ltr_features(
            judgments_df=judgments,
            bm25_index=bm25_index,
        )

        validate_feature_dataset(
            judgments,
            features,
            split_name,
        )

        feature_datasets[split_name] = features

        summary_rows.append(
            {
                "split": split_name,
                "queries": features["query_id"].nunique(),
                "rows": len(features),
                "mean_bm25": features["bm25_score"].mean(),
                "mean_title_overlap": (
                    features["title_token_overlap"].mean()
                ),
                "mean_query_coverage": (
                    features["query_coverage"].mean()
                ),
            }
        )

    # Confirm every cleaned WANDS judgment was processed.
    total_rows = sum(
        len(data) for data in feature_datasets.values()
    )

    total_queries = sum(
        data["query_id"].nunique()
        for data in feature_datasets.values()
    )

    if total_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, found {total_rows}"
        )

    if total_queries != EXPECTED_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_QUERIES} queries, "
            f"found {total_queries}"
        )

    # Verify query-level separation again.
    query_sets = {
        name: set(feature_datasets[name]["query_id"])
        for name in SPLIT_NAMES
    }

    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1:]:
            if query_sets[first] & query_sets[second]:
                raise ValueError(
                    f"Query leakage: {first} vs {second}"
                )

    # Only save after every partition passes validation.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name, data in feature_datasets.items():
        output_path = OUTPUT_DIR / f"{split_name}_features.csv"

        data.to_csv(
            output_path,
            index=False,
        )

        print(f"Saved {output_path}")

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "feature_summary.csv",
        index=False,
    )

    print("\nFeature generation summary:")
    print(summary.to_string(index=False))

    print("\nTotal feature rows:", total_rows)
    print("Total queries:", total_queries)
    print("\nWANDS feature generation completed successfully.")


if __name__ == "__main__":
    main()
