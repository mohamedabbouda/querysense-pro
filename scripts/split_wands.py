
"""Build reproducible train/validation/test splits from WANDS."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from querysense.data.wands_adapter import load_wands_products
from querysense.data.wands_relevance import load_wands_relevance
from querysense.ranking.wands_split import (
    normalize_query_for_split,
    split_wands_relevance,
    summarize_wands_splits,
)

WANDS_DIR = Path("../WANDS/dataset")
OUTPUT_DIR = Path("data/processed/wands")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

SPLIT_NAMES = ("train", "validation", "test")


def validate_splits(
    relevance: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> None:
    """Check that splits preserve data without query leakage."""

    # All original judgments must be assigned exactly once.
    total_rows = sum(len(part) for part in splits.values())

    if total_rows != len(relevance):
        raise ValueError(
            f"Judgment count mismatch: {total_rows} != {len(relevance)}"
        )

    query_sets = {
        name: set(splits[name]["query_id"])
        for name in SPLIT_NAMES
    }

    # A query ID must not appear in multiple splits.
    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1:]:
            overlap = query_sets[first] & query_sets[second]

            if overlap:
                raise ValueError(
                    f"Query leakage between {first} and {second}: "
                    f"{len(overlap)} queries"
                )

    # Normalized query text must not cross split boundaries.
    normalized_sets = {}

    for name in SPLIT_NAMES:
        normalized_sets[name] = set(
            splits[name]["query"].map(normalize_query_for_split)
        )

    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1:]:
            overlap = (
                normalized_sets[first] & normalized_sets[second]
            )

            if overlap:
                raise ValueError(
                    f"Normalized query leakage: {first} vs {second}"
                )

    # Every cleaned query-product judgment must be preserved.
    keys = ["query_id", "product_id"]

    combined = pd.concat(
        [splits[name] for name in SPLIT_NAMES],
        ignore_index=True,
    )

    if combined.duplicated(keys).any():
        raise ValueError(
            "Duplicate query-product pairs across splits"
        )

    original_pairs = set(
        relevance[keys].itertuples(index=False, name=None)
    )

    split_pairs = set(
        combined[keys].itertuples(index=False, name=None)
    )

    if original_pairs != split_pairs:
        raise ValueError(
            "Split query-product pairs differ from original data"
        )

    # Ensure each partition contains at least one query.
    for name in SPLIT_NAMES:
        if splits[name]["query_id"].nunique() == 0:
            raise ValueError(f"Empty query partition: {name}")

    print("\nAll split integrity checks passed.")


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save judgment partitions and query assignments."""

    output_dir.mkdir(parents=True, exist_ok=True)

    query_assignments = []

    for name in SPLIT_NAMES:
        data = splits[name]

        output_path = output_dir / f"{name}.csv"
        data.to_csv(output_path, index=False)

        # Save which queries belong to which partition.
        assignments = (
            data[["query_id", "query"]]
            .drop_duplicates()
            .copy()
        )

        assignments["split"] = name
        query_assignments.append(assignments)

        print(f"Saved {output_path}: {len(data)} judgments")

    assignments_df = pd.concat(
        query_assignments,
        ignore_index=True,
    )

    assignments_df = assignments_df.sort_values(
        ["split", "query_id"]
    ).reset_index(drop=True)

    assignments_path = output_dir / "query_assignments.csv"

    assignments_df.to_csv(
        assignments_path,
        index=False,
    )

    print(f"Saved {assignments_path}")


def main() -> None:
    print("Loading WANDS products...")

    products = load_wands_products(
        WANDS_DIR / "product.csv"
    )

    print(f"Loaded {len(products):,} products")

    print("\nLoading and cleaning relevance judgments...")

    relevance = load_wands_relevance(
        query_path=WANDS_DIR / "query.csv",
        label_path=WANDS_DIR / "label.csv",
        products_df=products,
        duplicate_policy="exclude_conflicts",
    )

    print(f"Cleaned judgments: {len(relevance):,}")
    print(f"Unique queries: {relevance['query_id'].nunique()}")

    print("\nSplitting WANDS by normalized queries...")

    splits = split_wands_relevance(
        relevance,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    validate_splits(relevance, splits)

    summary = summarize_wands_splits(splits)

    print("\nWANDS split summary:")
    print(summary.to_string(index=False))

    # Verify the dataset totals observed during inspection.
    assert len(relevance) == 231_859
    assert relevance["query_id"].nunique() == 480

    assert summary.set_index("split")["queries"].to_dict() == {
        "train": 336,
        "validation": 72,
        "test": 72,
    }

    save_splits(splits, OUTPUT_DIR)

    summary.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False,
    )

    metadata = {
        "dataset": "WANDS",
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "duplicate_policy": "exclude_conflicts",
        "split_unit": "normalized_query",
        "total_judgments": len(relevance),
        "total_queries": int(relevance["query_id"].nunique()),
        "relevance_grades": {
            "Irrelevant": 0,
            "Partial": 1,
            "Exact": 2,
        },
    }

    metadata_path = OUTPUT_DIR / "split_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved {metadata_path}")
    print("\nWANDS dataset splitting complete.")


if __name__ == "__main__":
    main()
