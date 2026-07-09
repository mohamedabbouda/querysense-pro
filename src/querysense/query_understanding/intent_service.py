from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor
from querysense.query_understanding.intent_predictor import HybridIntentPredictor, IntentPrediction
from querysense.training.train_intent import load_intent_classifier


@dataclass(frozen=True)
class IntentServiceConfig:
    """Configuration for intent prediction service."""

    model_path: str | Path
    products_path: str | Path


class IntentPredictionService:
    """Reusable service for intent prediction."""

    def __init__(self, config: IntentServiceConfig) -> None:
        self.config = config

        products_df = pd.read_parquet(config.products_path)
        entity_extractor = RuleBasedEntityExtractor.from_products(products_df)
        model = load_intent_classifier(config.model_path)

        self.predictor = HybridIntentPredictor(
            model=model,
            entity_extractor=entity_extractor,
        )

    def predict(self, query: str) -> IntentPrediction:
        """Predict intent for a single query."""
        return self.predictor.predict(query)

    def predict_batch(self, queries: list[str]) -> list[IntentPrediction]:
        """Predict intents for multiple queries."""
        return self.predictor.predict_batch(queries)