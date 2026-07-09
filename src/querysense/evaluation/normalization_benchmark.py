from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from querysense.query_understanding.normalizer import QueryNormalizer


@dataclass(frozen=True)
class NormalizationBenchmarkResult:
    """Result summary for noisy query normalization benchmark."""

    total_examples: int
    correct_examples: int
    accuracy: float
    results: pd.DataFrame


def run_normalization_benchmark(
    benchmark_path: str | Path,
    normalizer: QueryNormalizer | None = None,
) -> NormalizationBenchmarkResult:
    """Evaluate query normalization on a noisy query benchmark."""
    path = Path(benchmark_path)

    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    normalizer = normalizer or QueryNormalizer()
    df = pd.read_csv(path)

    required_columns = {"raw_query", "expected_normalized_query"}
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required benchmark columns: {missing_columns}")

    predictions: list[str] = []
    is_correct: list[bool] = []

    for _, row in df.iterrows():
        result = normalizer.normalize(row["raw_query"])
        prediction = result.normalized_query
        expected = str(row["expected_normalized_query"]).strip().lower()

        predictions.append(prediction)
        is_correct.append(prediction == expected)

    results_df = df.copy()
    results_df["predicted_normalized_query"] = predictions
    results_df["is_correct"] = is_correct

    total_examples = len(results_df)
    correct_examples = int(results_df["is_correct"].sum())
    accuracy = correct_examples / total_examples if total_examples else 0.0

    return NormalizationBenchmarkResult(
        total_examples=total_examples,
        correct_examples=correct_examples,
        accuracy=accuracy,
        results=results_df,
    )
    