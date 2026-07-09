from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class IntentDatasetSplit:
    """Train/test split for intent classification."""

    train_texts: list[str]
    test_texts: list[str]
    train_labels: list[str]
    test_labels: list[str]


REQUIRED_INTENT_COLUMNS = ["normalized_query", "intent"]


def prepare_intent_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean intent classification dataset.

    Uses normalized_query as model input and intent as target label.
    """
    missing_columns = sorted(set(REQUIRED_INTENT_COLUMNS) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required intent dataset columns: {missing_columns}")

    dataset = df[REQUIRED_INTENT_COLUMNS].copy()
    dataset = dataset.dropna(subset=REQUIRED_INTENT_COLUMNS)
    dataset["normalized_query"] = dataset["normalized_query"].astype(str).str.strip()
    dataset["intent"] = dataset["intent"].astype(str).str.strip()

    dataset = dataset[
        (dataset["normalized_query"] != "")
        & (dataset["intent"] != "")
    ]

    return dataset.reset_index(drop=True)


def split_intent_dataset(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> IntentDatasetSplit:
    """Split intent dataset into train and test sets."""
    train_df, test_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_seed,
        stratify=dataset["intent"],
    )

    return IntentDatasetSplit(
        train_texts=train_df["normalized_query"].tolist(),
        test_texts=test_df["normalized_query"].tolist(),
        train_labels=train_df["intent"].tolist(),
        test_labels=test_df["intent"].tolist(),
    )