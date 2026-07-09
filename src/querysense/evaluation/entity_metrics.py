from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from querysense.utils.text import normalize_text_basic


@dataclass(frozen=True)
class EntityFieldMetric:
    """Evaluation result for one entity field."""

    field_name: str
    total_examples: int
    evaluated_examples: int
    correct_examples: int
    accuracy: float


@dataclass(frozen=True)
class EntityEvaluationResult:
    """Full entity extraction evaluation result."""

    field_metrics: list[EntityFieldMetric]
    overall_accuracy: float
    results: pd.DataFrame


ENTITY_FIELDS = [
    "brand",
    "category",
    "subcategory",
    "color",
    "size",
    "gender",
    "condition",
    "max_price",
]


def evaluate_entity_extraction(
    df: pd.DataFrame,
    entity_fields: list[str] | None = None,
) -> EntityEvaluationResult:
    """Evaluate extracted entity columns against expected entity columns.

    Expected columns:
        brand, color, size, ...

    Predicted columns:
        extracted_brand, extracted_color, extracted_size, ...
    """
    fields = entity_fields or ENTITY_FIELDS
    results_df = df.copy()
    field_metrics: list[EntityFieldMetric] = []

    all_correct_values: list[bool] = []

    for field in fields:
        expected_column = field
        predicted_column = f"extracted_{field}"

        if expected_column not in results_df.columns:
            raise ValueError(f"Missing expected column: {expected_column}")

        if predicted_column not in results_df.columns:
            raise ValueError(f"Missing predicted column: {predicted_column}")

        comparison_column = f"{field}_is_correct"
        evaluated_column = f"{field}_is_evaluated"


        expected_values = results_df[expected_column].apply(_normalize_entity_value)
        predicted_values = results_df[predicted_column].apply(_normalize_entity_value)

        # We only evaluate rows where the expected entity is present.
        # Example: if expected brand is empty, we do not punish extracted_brand=None.
        evaluated_mask = expected_values.notna()

        correctness = (expected_values == predicted_values) & evaluated_mask
        results_df[evaluated_column] = evaluated_mask
        results_df[comparison_column] = correctness

        evaluated_examples = int(evaluated_mask.sum())
        correct_examples = int(correctness.sum())
        accuracy = correct_examples / evaluated_examples if evaluated_examples else 0.0

        field_metrics.append(
            EntityFieldMetric(
                field_name=field,
                total_examples=len(results_df),
                evaluated_examples=evaluated_examples,
                correct_examples=correct_examples,
                accuracy=accuracy,
            )
        )

        all_correct_values.extend(correctness[evaluated_mask].tolist())

    overall_accuracy = (
        sum(all_correct_values) / len(all_correct_values) if all_correct_values else 0.0
    )

    return EntityEvaluationResult(
        field_metrics=field_metrics,
        overall_accuracy=overall_accuracy,
        results=results_df,
    )


def _normalize_entity_value(value: object) -> str | float | None:
    """Normalize entity values for fair comparison."""
    if pd.isna(value):
        return None

    if isinstance(value, float | int):
        return float(value)

    text = str(value).strip()

    if not text:
        return None

    # Normalize strings the same way as query/catalog matching.
    return normalize_text_basic(text)