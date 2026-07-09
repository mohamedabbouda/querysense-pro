from __future__ import annotations

from dataclasses import dataclass, field

from querysense.utils.text import normalize_text_basic

DEFAULT_TERM_NORMALIZATION_MAP = {
    # Typos
    "iphon": "iphone",
    "iphonee": "iphone",
    "blak": "black",
    "whte": "white",
    "shoez": "shoes",
    "headfone": "headphones",
    "headfones": "headphones",
    "lapto": "laptop",
    "labtop": "laptop",
    "keybord": "keyboard",
    "wirless": "wireless",
    "camra": "camera",
    # Common variants
    "mens": "men",
    "man's": "men",
    "womens": "women",
    "woman": "women",
    "ladies": "women",
    # Synonyms / search variants
    "sneaker": "shoes",
    "sneakers": "shoes",
    "trainers": "shoes",
    "notebook": "laptop",
    "cellphone": "phone",
    "mobile": "phone",
    "smartphone": "phone",
    "earphones": "headphones",
    "earbuds": "headphones",
    "qwertz": "german",
}


@dataclass(frozen=True)
class NormalizedQuery:
    """Structured output of query normalization."""

    raw_query: str
    normalized_query: str
    tokens: list[str]
    corrected_tokens: list[str]
    corrections: dict[str, str] = field(default_factory=dict)


class QueryNormalizer:
    """Normalize raw user search queries into a cleaner representation."""

    def __init__(self, term_map: dict[str, str] | None = None) -> None:
        self.term_map = term_map or DEFAULT_TERM_NORMALIZATION_MAP

    def normalize(self, query: str) -> NormalizedQuery:
        """Normalize a raw search query.

        Args:
            query: Raw user query.

        Returns:
            Structured normalized query object.

        Raises:
            ValueError: If query is empty or not a string.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        normalized_query = normalize_text_basic(query)

        if not normalized_query:
            raise ValueError("Query cannot be empty.")

        tokens = normalized_query.split()
        corrected_tokens, corrections = self._normalize_terms(tokens)
        corrected_query = " ".join(corrected_tokens)

        return NormalizedQuery(
            raw_query=query,
            normalized_query=corrected_query,
            tokens=tokens,
            corrected_tokens=corrected_tokens,
            corrections=corrections,
        )

    def _normalize_terms(self, tokens: list[str]) -> tuple[list[str], dict[str, str]]:
        """Apply typo and synonym normalization to query tokens."""
        corrected_tokens: list[str] = []
        corrections: dict[str, str] = {}

        for token in tokens:
            corrected_token = self.term_map.get(token, token)
            corrected_tokens.append(corrected_token)

            if corrected_token != token:
                corrections[token] = corrected_token

        return corrected_tokens, corrections