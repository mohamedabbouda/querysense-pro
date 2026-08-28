from __future__ import annotations

from querysense.evaluation.search_relevance import SearchRelevanceExample
from querysense.query_understanding.entities import ExtractedEntities
from querysense.ranking.ltr_dataset import (
    build_ltr_feature_dataset,
    build_ltr_feature_row,
)
from querysense.retrieval.search_result import ProductSearchResponse, ProductSearchResult


def test_build_ltr_feature_row_creates_expected_features() -> None:
    response = ProductSearchResponse(
        query="nike black running shoes",
        normalized_query="nike black running shoes",
        expanded_query="nike black running shoes",
        expanded_terms=[],
        intent="product_search",
        entities=ExtractedEntities(
            brand="nike",
            subcategory="shoes",
            color="black",
        ),
        recommended_filters=[],
        results=[],
    )
    result = ProductSearchResult(
        product_id="P001",
        title="Nike Men's Black Running Shoes Size 44",
        brand="nike",
        category="fashion",
        subcategory="shoes",
        color="black",
        size="44",
        gender="men",
        condition="new",
        price=89.99,
        currency="EUR",
        score=5.5,
        bm25_score=3.2,
        semantic_score=0.8,
        match_reasons=["brand", "subcategory", "color", "bm25", "semantic"],
    )

    row = build_ltr_feature_row(
        response=response,
        result=result,
        label=1,
    )

    assert row.query == "nike black running shoes"
    assert row.product_id == "P001"
    assert row.label == 1
    assert row.manual_score == 5.5
    assert row.bm25_score == 3.2
    assert row.semantic_score == 0.8
    assert row.brand_match == 1
    assert row.subcategory_match == 1
    assert row.color_match == 1
    assert row.category_match == 0
    assert row.has_bm25_match == 1
    assert row.has_semantic_match == 1
    assert row.title_token_overlap_count == 4
    assert row.title_token_overlap_ratio == 1.0


def test_build_ltr_feature_row_handles_missing_entity_matches() -> None:
    response = ProductSearchResponse(
        query="running shoes",
        normalized_query="running shoes",
        expanded_query="running shoes",
        expanded_terms=[],
        intent="product_search",
        entities=ExtractedEntities(),
        recommended_filters=[],
        results=[],
    )
    result = ProductSearchResult(
        product_id="P002",
        title="Adidas White Sneakers",
        brand="adidas",
        category="fashion",
        subcategory="shoes",
        color="white",
        size="39",
        gender="women",
        condition="new",
        price=74.99,
        currency="EUR",
        score=1.0,
        bm25_score=0.0,
        semantic_score=0.0,
        match_reasons=[],
    )

    row = build_ltr_feature_row(
        response=response,
        result=result,
        label=0,
    )

    assert row.label == 0
    assert row.brand_match == 0
    assert row.category_match == 0
    assert row.subcategory_match == 0
    assert row.color_match == 0
    assert row.has_bm25_match == 0
    assert row.has_semantic_match == 0
    assert row.title_token_overlap_count == 0
    assert row.title_token_overlap_ratio == 0.0


class FakeSearchService:
    def search(self, query: str) -> ProductSearchResponse:
        result = ProductSearchResult(
            product_id="P001",
            title="Nike Black Running Shoes",
            brand="nike",
            category="fashion",
            subcategory="shoes",
            color="black",
            size="44",
            gender="men",
            condition="new",
            price=89.99,
            currency="EUR",
            score=5.0,
            bm25_score=2.0,
            semantic_score=0.0,
            match_reasons=["brand", "bm25"],
        )

        return ProductSearchResponse(
            query=query,
            normalized_query=query,
            expanded_query=query,
            expanded_terms=[],
            intent="product_search",
            entities=ExtractedEntities(brand="nike"),
            recommended_filters=[],
            results=[result],
        )


def test_build_ltr_feature_dataset_labels_relevant_products() -> None:
    relevance_examples = [
        SearchRelevanceExample(
            query="nike shoes",
            relevant_product_ids={"P001"},
        )
    ]

    dataset = build_ltr_feature_dataset(
        search_service=FakeSearchService(),  # type: ignore[arg-type]
        relevance_examples=relevance_examples,
    )

    assert len(dataset) == 1
    assert dataset.loc[0, "query"] == "nike shoes"
    assert dataset.loc[0, "product_id"] == "P001"
    assert dataset.loc[0, "label"] == 1
    assert dataset.loc[0, "brand_match"] == 1