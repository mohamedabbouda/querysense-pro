from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LTR_FEATURE_COLUMNS = [
    "manual_score",
    "bm25_score",
    "semantic_score",
    "price",
    "brand_match",
    "category_match",
    "subcategory_match",
    "color_match",
    "size_match",
    "gender_match",
    "condition_match",
    "has_bm25_match",
    "has_semantic_match",
    "title_token_overlap_count",
    "title_token_overlap_ratio",
]


@dataclass(frozen=True)
class LTRTrainingResult:
    """Result from training a pointwise LTR model."""

    model: Pipeline
    feature_columns: list[str]
    num_rows: int
    num_positive_rows: int
    num_negative_rows: int


def train_pointwise_ltr_model(
    dataset: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> LTRTrainingResult:
    """Train a pointwise Logistic Regression LTR baseline."""
    selected_features = feature_columns or LTR_FEATURE_COLUMNS

    _validate_ltr_dataset(
        dataset=dataset,
        feature_columns=selected_features,
    )

    x_train = dataset[selected_features]
    y_train = dataset["label"].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    return LTRTrainingResult(
        model=model,
        feature_columns=selected_features,
        num_rows=len(dataset),
        num_positive_rows=int(y_train.sum()),
        num_negative_rows=int((y_train == 0).sum()),
    )


def predict_ltr_scores(
    model: Pipeline,
    dataset: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.Series:
    """Predict relevance probabilities for query-product rows."""
    selected_features = feature_columns or LTR_FEATURE_COLUMNS

    _validate_feature_columns(
        dataset=dataset,
        feature_columns=selected_features,
    )

    probabilities = model.predict_proba(dataset[selected_features])[:, 1]

    return pd.Series(probabilities, index=dataset.index, name="ltr_score")


def _validate_ltr_dataset(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    if dataset.empty:
        raise ValueError("LTR dataset must not be empty")

    if "label" not in dataset.columns:
        raise ValueError("LTR dataset must contain a label column")

    labels = set(dataset["label"].astype(int).unique())

    if not labels.issubset({0, 1}):
        raise ValueError("LTR labels must be binary values: 0 or 1")

    if labels != {0, 1}:
        raise ValueError("LTR dataset must contain both positive and negative rows")

    _validate_feature_columns(
        dataset=dataset,
        feature_columns=feature_columns,
    )


def _validate_feature_columns(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    missing_columns = set(feature_columns) - set(dataset.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing LTR feature columns: {missing}")