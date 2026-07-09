from __future__ import annotations

from querysense.config import load_project_configs
from querysense.query_understanding.intent_service import (
    IntentPredictionService,
    IntentServiceConfig,
)


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]
    training_config = configs["training"]

    intent_config = training_config["intent_classifier"]

    service = IntentPredictionService(
        IntentServiceConfig(
            model_path=intent_config["model_output_path"],
            products_path=base_config["files"]["processed_products_parquet"],
        )
    )

    example_queries = [
        "sony black headphones",
        "sony",
        "headphones",
        "new sony headphones",
        "sony headphones under 300",
        "cheap laptop",
        "iphon blak",
    ]

    print("Intent predictions")
    print()

    for query in example_queries:
        prediction = service.predict(query)

        print(f"Query: {query}")
        print(f"Normalized query: {prediction.normalized_query}")
        print(f"Intent: {prediction.intent}")
        print(f"Source: {prediction.source}")
        print(f"Model intent: {prediction.model_intent}")
        print("-" * 50)


if __name__ == "__main__":
    main()