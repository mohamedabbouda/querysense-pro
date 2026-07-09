from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from querysense.data.intent_dataset import IntentDatasetSplit


@dataclass(frozen=True)
class TrainedIntentClassifier:
    """Trained intent classifier and metadata."""

    pipeline: Pipeline
    labels: list[str]


def train_intent_classifier(
    split: IntentDatasetSplit,
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    random_seed: int = 42,
) -> TrainedIntentClassifier:
    """Train a TF-IDF + Logistic Regression intent classifier."""
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(split.train_texts, split.train_labels)

    labels = sorted(set(split.train_labels))

    return TrainedIntentClassifier(
        pipeline=pipeline,
        labels=labels,
    )


def save_intent_classifier(
    trained_model: TrainedIntentClassifier,
    model_output_path: str | Path,
    label_output_path: str | Path,
) -> None:
    """Save trained intent classifier and label list."""
    model_path = Path(model_output_path)
    labels_path = Path(label_output_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained_model.pipeline, model_path)

    with labels_path.open("w", encoding="utf-8") as file:
        json.dump(trained_model.labels, file, indent=2)


def load_intent_classifier(model_path: str | Path) -> Pipeline:
    """Load trained intent classifier pipeline."""
    return joblib.load(model_path)


    