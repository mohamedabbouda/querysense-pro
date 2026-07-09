from __future__ import annotations

from querysense.utils.text import (
    normalize_text_basic,
    normalize_whitespace,
    remove_extra_punctuation,
    strip_accents,
    to_lowercase,
)


def test_to_lowercase() -> None:
    assert to_lowercase("NIKE Shoes") == "nike shoes"


def test_strip_accents() -> None:
    assert strip_accents("café") == "cafe"


def test_normalize_whitespace() -> None:
    assert normalize_whitespace(" nike    shoes   black ") == "nike shoes black"


def test_remove_extra_punctuation() -> None:
    assert remove_extra_punctuation("nike!!! shoes???") == "nike    shoes   "


def test_normalize_text_basic() -> None:
    assert normalize_text_basic("  NIKE!!!   Black Shoes??? ") == "nike black shoes"


def test_normalize_text_basic_keeps_price_symbols() -> None:
    assert normalize_text_basic("Laptop under €700!!!") == "laptop under €700"