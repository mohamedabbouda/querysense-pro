from __future__ import annotations

import pandas as pd
import pytest

from querysense.evaluation.normalization_benchmark import run_normalization_benchmark


def test_run_normalization_benchmark_successfully() -> None:
    result = run_normalization_benchmark("data/samples/noisy_queries.csv")

    assert result.total_examples == 10
    assert result.correct_examples >= 9
    assert result.accuracy >= 0.9
    assert isinstance(result.results, pd.DataFrame)
    assert "predicted_normalized_query" in result.results.columns
    assert "is_correct" in result.results.columns


def test_run_normalization_benchmark_missing_file_raises_error() -> None:
    with pytest.raises(FileNotFoundError):
        run_normalization_benchmark("data/samples/missing_noisy_queries.csv")


def test_run_normalization_benchmark_missing_columns_raises_error(tmp_path) -> None:
    broken_file = tmp_path / "broken_benchmark.csv"
    broken_file.write_text("query,expected\niphon,iphone\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required benchmark columns"):
        run_normalization_benchmark(broken_file)