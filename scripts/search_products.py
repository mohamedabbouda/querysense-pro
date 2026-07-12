from __future__ import annotations

from querysense.config import load_project_configs
from querysense.retrieval.search_service import ProductSearchService, ProductSearchServiceConfig


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]
    training_config = configs["training"]
    intent_config = training_config["intent_classifier"]

    service = ProductSearchService(
        ProductSearchServiceConfig(
            model_path=intent_config["model_output_path"],
            products_path=base_config["files"]["processed_products_parquet"],
            max_results=5,
        )
    )

    example_queries = [
        "sony black headphones under 300",
        "nike black shoes",
        "iphone black",
        "cheap laptop",
        "white shoes for women",
    ]

    for query in example_queries:
        response = service.search(query)

        print("=" * 80)
        print(f"Query: {response.query}")
        print(f"Normalized query: {response.normalized_query}")
        print(f"Intent: {response.intent}")
        print("Results:")

        if not response.results:
            print("No results found.")
            continue

        for result in response.results:
            print(
                f"- {result.title} | {result.brand} | "
                f"{result.price:.2f} {result.currency} | "
                f"score={result.score:.2f} | reasons={result.match_reasons}"
            )


if __name__ == "__main__":
    main()