from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from querysense.query_understanding.entities import ExtractedEntities
from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor
from querysense.query_understanding.normalizer import QueryNormalizer


@dataclass(frozen=True)
class IntentPrediction:
    """Intent prediction result."""

    intent: str
    source: str
    normalized_query: str
    model_intent: str | None = None


class HybridIntentPredictor:
    """Hybrid intent predictor using deterministic rules plus an ML classifier."""

    def __init__(
        self,
        model: Pipeline,
        entity_extractor: RuleBasedEntityExtractor,
        normalizer: QueryNormalizer | None = None,
    ) -> None:
        self.model = model
        self.entity_extractor = entity_extractor
        self.normalizer = normalizer or QueryNormalizer()

    def predict(self, query: str) -> IntentPrediction:
        """Predict intent for a single query."""
        normalized = self.normalizer.normalize(query)
        normalized_query = normalized.normalized_query

        entities = self.entity_extractor.extract(query)
        model_intent = str(self.model.predict([normalized_query])[0])

        rule_intent = self._predict_with_rules(
            normalized_query=normalized_query,
            entities=entities,
        )

        if rule_intent is not None:
            return IntentPrediction(
                intent=rule_intent,
                source="rule",
                normalized_query=normalized_query,
                model_intent=model_intent,
            )

        return IntentPrediction(
            intent=model_intent,
            source="model",
            normalized_query=normalized_query,
            model_intent=model_intent,
        )

    def predict_batch(self, queries: list[str]) -> list[IntentPrediction]:
        """Predict intents for multiple queries."""
        return [self.predict(query) for query in queries]

    def _predict_with_rules(
        self,
        normalized_query: str,
        entities: ExtractedEntities,
    ) -> str | None:
        tokens = normalized_query.split()
        if entities.max_price is not None or entities.min_price is not None:
            return "price_search"
        if (
            entities.condition is not None
            or entities.size is not None
            or entities.gender is not None
            ):
            return "filtered_product_search"
        if (
            entities.brand is not None
            and entities.subcategory is not None
            and entities.color is not None
            ):
            return "product_search"
        if entities.color is not None:
            return "filtered_product_search"
        if len(tokens) == 1:
            token = tokens[0]

            if entities.brand == token:
                return "brand_search"

            if entities.subcategory == token or entities.category == token:
                return "category_search"

        if (
            entities.brand is not None
            and entities.subcategory is not None
            and entities.color is None
            and entities.size is None
            and entities.condition is None
        ):
            return "product_search"

        return None