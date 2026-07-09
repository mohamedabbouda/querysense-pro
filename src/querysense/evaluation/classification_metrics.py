from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass(frozen=True)
class ClassificationEvaluationResult:
    """Evaluation result for a classification model."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    classification_report_text: str
    confusion_matrix: pd.DataFrame
    predictions: pd.DataFrame


def evaluate_classifier(
    true_labels: list[str],
    predicted_labels: list[str],
    labels: list[str] | None = None,
) -> ClassificationEvaluationResult:
    """Evaluate a classification model using standard classification metrics."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length.")

    if not true_labels:
        raise ValueError("Cannot evaluate an empty label list.")

    ordered_labels = labels or sorted(set(true_labels) | set(predicted_labels))

    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=ordered_labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=ordered_labels,
        zero_division=0,
    )
    
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=ordered_labels,
    )

    confusion_matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in ordered_labels],
        columns=[f"pred_{label}" for label in ordered_labels],
    )

    predictions_df = pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "is_correct": [
                true_label == predicted_label
                for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True)
            ],
        }
    )

    return ClassificationEvaluationResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        classification_report_text=report,
        confusion_matrix=confusion_matrix_df,
        predictions=predictions_df,
    )