from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from querysense.query_understanding.entities import ExtractedEntities
from querysense.query_understanding.normalizer import QueryNormalizer
from querysense.utils.text import normalize_text_basic

DEFAULT_PRODUCT_ALIAS_TO_BRAND = {
    "iphone": "apple",
    "ipad": "apple",
    "macbook": "apple",
    "magic keyboard": "apple",
    "galaxy": "samsung",
    "thinkpad": "lenovo",
    "xps": "dell",
    "wh-1000xm4": "sony",
    "mx master": "logitech",
    "hero11": "gopro",
}
DEFAULT_PRODUCT_TYPE_TO_SUBCATEGORY = {
    "shoe": "shoes",
    "shoes": "shoes",
    "sneaker": "shoes",
    "sneakers": "shoes",
    "trainer": "shoes",
    "trainers": "shoes",
    "phone": "smartphones",
    "smartphone": "smartphones",
    "iphone": "smartphones",
    "galaxy": "smartphones",
    "headphone": "headphones",
    "headphones": "headphones",
    "earphone": "headphones",
    "earphones": "headphones",
    "earbud": "headphones",
    "earbuds": "headphones",
    "laptop": "laptops",
    "notebook": "laptops",
    "camera": "cameras",
    "cameras": "cameras",
    "jacket": "jackets",
    "coat": "coats",
    "dress": "dresses",
    "jeans": "jeans",
    "desk": "furniture",
    "chair": "furniture",
    "accessory": "accessories",
    "keyboard": "accessories",
    "mouse": "accessories",
    "air fryer": "kitchen",
    "vacuum": "appliances",
    "vacuum cleaner": "appliances",
    "smartwatch": "wearables",
    "watch": "wearables",
}

PRICE_INTENT_TERMS = {
    "cheap": "cheap",
    "budget": "cheap",
    "affordable": "cheap",
    "low-cost": "cheap",
    "expensive": "expensive",
    "premium": "expensive",
    "luxury": "expensive",
}

MAX_PRICE_PATTERNS = [
    re.compile(r"(?:under|below|less than|max|maximum|up to)\s*[€$£]?\s*(\d+(?:\.\d+)?)"),
    re.compile(r"[€$£]?\s*(\d+(?:\.\d+)?)\s*(?:or less|and below)"),
]

MIN_PRICE_PATTERNS = [
    re.compile(r"(?:over|above|more than|min|minimum|from)\s*[€$£]?\s*(\d+(?:\.\d+)?)"),
    re.compile(r"[€$£]?\s*(\d+(?:\.\d+)?)\s*(?:or more|and above)"),
]


@dataclass(frozen=True)
class EntityVocabulary:
    """Vocabulary used by the rule-based entity extractor."""

    brands: set[str]
    categories: set[str]
    subcategories: set[str]
    colors: set[str]
    sizes: set[str]
    genders: set[str]
    conditions: set[str]
    product_alias_to_brand: dict[str, str] = field(default_factory=dict)
    product_type_to_subcategory: dict[str, str] = field(default_factory=dict)
    subcategory_to_category: dict[str, str] = field(default_factory=dict)
    
class RuleBasedEntityExtractor:
    """Extract query entities using catalog-derived vocabularies."""

    def __init__(
        self,
        vocabulary: EntityVocabulary,
        normalizer: QueryNormalizer | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.normalizer = normalizer or QueryNormalizer()

    @classmethod
    def from_products(cls, products_df: pd.DataFrame) -> RuleBasedEntityExtractor:
        """Create an extractor from a product catalog."""

        vocabulary = EntityVocabulary(
            brands=_build_vocab(products_df["brand"]),
            categories=_build_vocab(products_df["category"]),
            subcategories=_build_vocab(products_df["subcategory"]),
            colors=_build_vocab(products_df["color"]),
            sizes=_build_vocab(products_df["size"]),
            genders=_build_vocab(products_df["gender"]),
            conditions=_build_vocab(products_df["condition"]),
            product_alias_to_brand=DEFAULT_PRODUCT_ALIAS_TO_BRAND,
            product_type_to_subcategory=DEFAULT_PRODUCT_TYPE_TO_SUBCATEGORY,
            subcategory_to_category=_build_subcategory_to_category(products_df),
            

        )

        return cls(vocabulary=vocabulary)

    def extract(self, query: str) -> ExtractedEntities:
        """Extract entities from a raw query string."""
        normalized = self.normalizer.normalize(query)
        normalized_query = normalized.normalized_query
        tokens = normalized.corrected_tokens

        brand = _find_phrase_match(normalized_query, self.vocabulary.brands)
        alias_brand = _find_alias_brand(
            normalized_query=normalized_query,
            alias_to_brand=self.vocabulary.product_alias_to_brand,
        )

        if brand is None:
            brand = alias_brand

        category = _find_phrase_match(normalized_query, self.vocabulary.categories)
        subcategory = _find_phrase_match(normalized_query, self.vocabulary.subcategories)
        alias_subcategory = _find_alias_subcategory(
            normalized_query=normalized_query,
            product_type_to_subcategory=self.vocabulary.product_type_to_subcategory,
            )

        if subcategory is None:
            subcategory = alias_subcategory
        if category is None and subcategory is not None:
            category = self.vocabulary.subcategory_to_category.get(subcategory)    
        color = _find_token_match(tokens, self.vocabulary.colors)
        size = _find_size_match(tokens, self.vocabulary.sizes, normalized_query)
        gender = _find_token_match(tokens, self.vocabulary.genders)
        condition = _find_token_match(tokens, self.vocabulary.conditions)

        min_price = _extract_min_price(normalized_query)
        max_price = _extract_max_price(normalized_query)
        price_intent = _extract_price_intent(tokens)

        product_type = subcategory

        return ExtractedEntities(
            brand=brand,
            category=category,
            subcategory=subcategory,
            product_type=product_type,
            color=color,
            size=size,
            gender=gender,
            condition=condition,
            min_price=min_price,
            max_price=max_price,
            price_intent=price_intent,
        )


def _build_vocab(values: pd.Series) -> set[str]:
    """Build a normalized vocabulary from a pandas Series."""
    return {
        normalize_text_basic(str(value))
        for value in values.dropna().unique()
        if normalize_text_basic(str(value))
    }

def _build_subcategory_to_category(products_df: pd.DataFrame) -> dict[str, str]:
    """Build mapping from normalized subcategory to normalized category."""
    mapping: dict[str, str] = {}

    for _, row in products_df.iterrows():
        subcategory = normalize_text_basic(str(row["subcategory"]))
        category = normalize_text_basic(str(row["category"]))

        if subcategory and category:
            mapping[subcategory] = category

    return mapping


def _find_token_match(tokens: list[str], vocabulary: set[str]) -> str | None:
    """Find the first token that exists in a vocabulary."""
    for token in tokens:
        if token in vocabulary:
            return token
    return None


def _find_phrase_match(query: str, vocabulary: set[str]) -> str | None:
    """Find the longest vocabulary phrase contained in the query."""
    matches = [term for term in vocabulary if term in query]
    if not matches:
        return None

    return max(matches, key=len)


def _find_alias_brand(normalized_query: str, alias_to_brand: dict[str, str]) -> str | None:
    """Find a brand through product aliases.

    Example:
        iphone -> apple
        galaxy -> samsung
        thinkpad -> lenovo
    """
    matches = [brand for alias, brand in alias_to_brand.items() if alias in normalized_query]

    if not matches:
        return None

    return max(matches, key=len)


def _extract_max_price(normalized_query: str) -> float | None:
    """Extract maximum price constraint from a query."""
    for pattern in MAX_PRICE_PATTERNS:
        match = pattern.search(normalized_query)
        if match:
            return float(match.group(1))
    return None


def _extract_min_price(normalized_query: str) -> float | None:
    """Extract minimum price constraint from a query."""
    for pattern in MIN_PRICE_PATTERNS:
        match = pattern.search(normalized_query)
        if match:
            return float(match.group(1))
    return None


def _extract_price_intent(tokens: list[str]) -> str | None:
    """Extract qualitative price intent such as cheap or expensive."""
    for token in tokens:
        if token in PRICE_INTENT_TERMS:
            return PRICE_INTENT_TERMS[token]
    return None

def _find_alias_subcategory(
    normalized_query: str,
    product_type_to_subcategory: dict[str, str],
    ) -> str | None:
    """Find catalog subcategory through product type aliases.

    Example:
        laptop -> laptops
        phone -> smartphones
        camera -> cameras
    """
    matches = [
        subcategory
        for product_type, subcategory in product_type_to_subcategory.items()
        if product_type in normalized_query
    ]

    if not matches:
        return None

    return max(matches, key=len) 

def _find_size_match(
    tokens: list[str],
    vocabulary: set[str],
    normalized_query: str,
    ) -> str | None:
    """Find size entities while avoiding false positives from brand fragments.

    Single-letter sizes like s, m, l are extracted when:
    - the query contains an explicit size cue, like "size m"
    - or the query contains a clothing product context, like "dress m"
    """
    explicit_size_cues = {"size", "w", "l"}
    single_letter_sizes = {"s", "m", "l"}
    clothing_context_terms = {
        "dress",
        "dresses",
        "coat",
        "coats",
        "jacket",
        "jackets",
        "jeans",
    }

    has_clothing_context = any(term in normalized_query for term in clothing_context_terms)

    for index, token in enumerate(tokens):
        if token not in vocabulary:
            continue

        if token in single_letter_sizes:
            previous_token = tokens[index - 1] if index > 0 else ""
            next_token = tokens[index + 1] if index + 1 < len(tokens) else ""

            is_brand_fragment = (
                (previous_token == "h" and token == "m")
                or (previous_token == "levi" and token == "s")
            )

            if is_brand_fragment:
                continue

            if previous_token in explicit_size_cues or has_clothing_context:
                return token

            if next_token in explicit_size_cues:
                return token

            continue

        return token

    return None