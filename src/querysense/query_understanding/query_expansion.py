from __future__ import annotations

from dataclasses import dataclass

from querysense.utils.text import normalize_text_basic

DEFAULT_QUERY_EXPANSION_RULES: dict[str, list[str]] = {
    "noise blocking headset": [
        "wireless headphones",
        "noise cancelling headphones",
    ],
    "noise blocking headphones": [
        "noise cancelling headphones",
    ],
    "headset": [
        "headphones",
    ],
    "something for jogging": [
        "running shoes",
    ],
    "jogging": [
        "running shoes",
    ],
    "phone for daily use": [
        "smartphone",
        "iphone",
        "galaxy",
    ],
    "daily use phone": [
        "smartphone",
    ],
    "office work laptop": [
        "business laptop",
        "thinkpad",
        "xps",
    ],
    "work laptop": [
        "business laptop",
    ],
    "office furniture": [
        "desk",
        "office chair",
    ],
}


@dataclass(frozen=True)
class ExpandedQuery:
    """A query with expansion terms for retrieval."""

    original_query: str
    normalized_query: str
    expanded_terms: list[str]
    expanded_query: str


class QueryExpander:
    """Rule-based query expansion for product search."""

    def __init__(
        self,
        expansion_rules: dict[str, list[str]] | None = None,
    ) -> None:
        self.expansion_rules = expansion_rules or DEFAULT_QUERY_EXPANSION_RULES
        self.normalized_rules = {
            normalize_text_basic(pattern): [
                normalize_text_basic(term)
                for term in terms
            ]
            for pattern, terms in self.expansion_rules.items()
        }

    def expand(self, query: str) -> ExpandedQuery:
        """Expand a query with related search terms."""
        normalized_query = normalize_text_basic(query)
        expanded_terms: list[str] = []

        for pattern, terms in self.normalized_rules.items():
            if pattern in normalized_query:
                expanded_terms.extend(terms)

        deduplicated_terms = _deduplicate_terms(expanded_terms)

        expanded_query_parts = [
            normalized_query,
            *deduplicated_terms,
        ]

        return ExpandedQuery(
            original_query=query,
            normalized_query=normalized_query,
            expanded_terms=deduplicated_terms,
            expanded_query=" ".join(expanded_query_parts),
        )


def _deduplicate_terms(terms: list[str]) -> list[str]:
    seen_terms = set()
    deduplicated_terms = []

    for term in terms:
        if term not in seen_terms:
            seen_terms.add(term)
            deduplicated_terms.append(term)

    return deduplicated_terms