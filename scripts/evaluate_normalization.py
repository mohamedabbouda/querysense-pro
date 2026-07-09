from __future__ import annotations

from querysense.config import load_project_configs
from querysense.evaluation.normalization_benchmark import run_normalization_benchmark


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]

    benchmark_path = base_config["files"]["noisy_queries_csv"]
    result = run_normalization_benchmark(benchmark_path)

    print("Normalization benchmark")
    print(f"Total examples: {result.total_examples}")
    print(f"Correct examples: {result.correct_examples}")
    print(f"Accuracy: {result.accuracy:.2%}")
    print()
    print(
        result.results[
            [
                "raw_query",
                "expected_normalized_query",
                "predicted_normalized_query",
                "is_correct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()