from __future__ import annotations

from querysense.query_understanding.query_expansion import QueryExpander


def test_query_expander_adds_expansion_terms_for_vague_query() -> None:
    expander = QueryExpander()

    result = expander.expand("noise blocking headset")

    assert result.original_query == "noise blocking headset"
    assert result.normalized_query == "noise blocking headset"
    assert "wireless headphones" in result.expanded_terms
    assert "noise cancelling headphones" in result.expanded_terms
    assert "headphones" in result.expanded_terms
    assert "noise blocking headset" in result.expanded_query


def test_query_expander_handles_jogging_query() -> None:
    expander = QueryExpander()

    result = expander.expand("something for jogging")

    assert result.expanded_terms == ["running shoes"]
    assert result.expanded_query == "something for jogging running shoes"


def test_query_expander_returns_original_query_when_no_rule_matches() -> None:
    expander = QueryExpander()

    result = expander.expand("nike black shoes")

    assert result.expanded_terms == []
    assert result.expanded_query == "nike black shoes"


def test_query_expander_deduplicates_expansion_terms() -> None:
    expander = QueryExpander(
        expansion_rules={
            "headset": ["headphones", "wireless headphones"],
            "noise headset": ["headphones", "noise cancelling headphones"],
        }
    )

    result = expander.expand("noise headset")

    assert result.expanded_terms == [
        "headphones",
        "wireless headphones",
        "noise cancelling headphones",
    ]