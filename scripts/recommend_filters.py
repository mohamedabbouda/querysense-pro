from __future__ import annotations

from querysense.config import load_project_configs
from querysense.query_understanding.filter_service import (
    FilterRecommendationService,
    FilterRecommendationServiceConfig,
)


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]

    service = FilterRecommendationService(
        FilterRecommendationServiceConfig(
            products_path=base_config["files"]["processed_products_parquet"],
        )
    )

    example_queries = [
        "sony black headphones under 300",
        "nike black shoes",
        "cheap laptop",
        "new h&m coat",
        "white shoes for women",
        "iphon blak",
    ]

    print("Filter recommendations")
    print()

    for query in example_queries:
        response = service.recommend(query)

        print("=" * 80)
        print(f"Query: {response.query}")
        print(f"Normalized query: {response.normalized_query}")

        if not response.filters:
            print("No filters recommended.")
            continue

        for filter_ in response.filters:
            print(
                f"- {filter_.name}: {filter_.value} "
                f"(confidence={filter_.confidence}, source={filter_.source})"
            )


if __name__ == "__main__":
    main()