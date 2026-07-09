from __future__ import annotations

import pandas as pd

from querysense.config import load_project_configs
from querysense.evaluation.entity_metrics import evaluate_entity_extraction


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]

    dataset_path = base_config["files"]["query_understanding_dataset_csv"]

    df = pd.read_csv(dataset_path)
    result = evaluate_entity_extraction(df)

    print("Entity extraction evaluation")
    print(f"Dataset: {dataset_path}")
    print(f"Total rows: {len(df)}")
    print(f"Overall entity accuracy: {result.overall_accuracy:.2%}")
    print()

    metrics_df = pd.DataFrame(
        [
            {
                "field": metric.field_name,
                "total_examples": metric.total_examples,
                "evaluated_examples": metric.evaluated_examples,
                "correct_examples": metric.correct_examples,
                "accuracy": f"{metric.accuracy:.2%}",
            }
            for metric in result.field_metrics
        ]
    )

    print(metrics_df.to_string(index=False))

    print()
    print("Example mistakes:")
    mistake_columns = [
        "query",
        "brand",
        "extracted_brand",
        "subcategory",
        "extracted_subcategory",
        "color",
        "extracted_color",
        "size",
        "extracted_size",
        "max_price",
        "extracted_max_price",
    ]

    mistake_fields = [
    "brand",
    "subcategory",
    "color",
    "size",
    "max_price",
    ]
    mistake_flags = [
    f"{field}_is_correct" for field in mistake_fields
    ]

    evaluated_flags = [
    f"{field}_is_evaluated" for field in mistake_fields
    ]

    available_columns = [
    column
    for column in mistake_columns + mistake_flags + evaluated_flags
    if column in result.results.columns
    ]

    mistake_mask = False
    
    for field in mistake_fields:
        correct_column = f"{field}_is_correct"
        evaluated_column = f"{field}_is_evaluated"
        if correct_column in result.results.columns and evaluated_column in result.results.columns:
            field_mistake_mask = (
            result.results[evaluated_column]
            & ~result.results[correct_column]
            )
            mistake_mask = mistake_mask | field_mistake_mask
        mistakes_df = result.results[mistake_mask]


    if mistakes_df.empty:
        print("No mistakes found for selected fields.")
    else:
        print(mistakes_df[available_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()