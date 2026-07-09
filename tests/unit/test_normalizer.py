from __future__ import annotations

import pytest

from querysense.query_understanding.normalizer import QueryNormalizer


def test_query_normalizer_basic_query() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("  NIKE!!! Black   Shoes??? ")

    assert result.raw_query == "  NIKE!!! Black   Shoes??? "
    assert result.normalized_query == "nike black shoes"
    assert result.tokens == ["nike", "black", "shoes"]
    assert result.corrected_tokens == ["nike", "black", "shoes"]
    assert result.corrections == {}


def test_query_normalizer_keeps_numbers_and_price_symbols() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("Laptop under €700!!!")

    assert result.normalized_query == "laptop under €700"
    assert result.tokens == ["laptop", "under", "€700"]


def test_query_normalizer_handles_accents() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("café table")

    assert result.normalized_query == "cafe table"


def test_query_normalizer_corrects_typos() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("iphon blak")

    assert result.normalized_query == "iphone black"
    assert result.tokens == ["iphon", "blak"]
    assert result.corrected_tokens == ["iphone", "black"]
    assert result.corrections == {
        "iphon": "iphone",
        "blak": "black",
    }


def test_query_normalizer_maps_synonyms() -> None:
    normalizer = QueryNormalizer()

    result = normalizer.normalize("mens sneaker")

    assert result.normalized_query == "men shoes"
    assert result.corrections == {
        "mens": "men",
        "sneaker": "shoes",
    }


def test_query_normalizer_supports_custom_term_map() -> None:
    normalizer = QueryNormalizer(term_map={"macbook": "laptop"})

    result = normalizer.normalize("macbook")

    assert result.normalized_query == "laptop"
    assert result.corrections == {"macbook": "laptop"}


def test_query_normalizer_rejects_empty_query() -> None:
    normalizer = QueryNormalizer()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        normalizer.normalize("   !!!   ")


def test_query_normalizer_rejects_non_string_query() -> None:
    normalizer = QueryNormalizer()

    with pytest.raises(ValueError, match="Query must be a string"):
        normalizer.normalize(123)  # type: ignore[arg-type]