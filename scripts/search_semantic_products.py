from __future__ import annotations

from querysense.config import get_project_root
from querysense.retrieval.semantic_service import (
    SemanticSearchService,
    SemanticSearchServiceConfig,
)


def main() -> None:
    project_root = get_project_root()
    products_path = project_root / "data" / "processed" / "products.parquet"

    service = SemanticSearchService(
        SemanticSearchServiceConfig(
            products_path=products_path,
            max_results=3,
        )
    )

    queries = [
        "noise blocking headset",
        "something for jogging",
        "office work laptop",
        "phone for daily use",
    ]

    for query in queries:
        print("=" * 80)
        print(f"Query: {query}")

        results = service.search(query)

        for _, product in results.iterrows():
            print(
                f"- {product['title']} | "
                f"{product['brand']} | "
                f"{product['price']} {product['currency']} | "
                f"semantic_score={product['semantic_score']:.4f}"
            )


if __name__ == "__main__":
    main()