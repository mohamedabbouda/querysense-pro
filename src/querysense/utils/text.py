from __future__ import annotations

import re
import unicodedata

_WHITESPACE_PATTERN = re.compile(r"\s+")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s€$£.-]")


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def strip_accents(text: str) -> str:
    """Remove accents from text.

    Example:
        café -> cafe
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_whitespace(text: str) -> str:
    """Replace repeated whitespace with a single space."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_extra_punctuation(text: str) -> str:
    """Remove punctuation that is usually not useful for search queries.

    Keeps:
    - word characters
    - whitespace
    - currency symbols
    - dots and hyphens for prices/model names
    """
    return _PUNCTUATION_PATTERN.sub(" ", text)


def normalize_text_basic(text: str) -> str:
    """Apply basic reusable text normalization."""
    text = strip_accents(text)
    text = to_lowercase(text)
    text = remove_extra_punctuation(text)
    text = normalize_whitespace(text)
    return text