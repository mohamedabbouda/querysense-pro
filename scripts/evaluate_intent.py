from __future__ import annotations

import pandas as pd

from querysense.config import load_project_configs
from querysense.data.intent_dataset import prepare_intent_dataset, split_intent_dataset
from querysense.evaluation.classification_metrics import evaluate_classifier
from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor
from querysense.query_understanding.intent_predictor import HybridIntentPredictor
from querysense.training.train_intent import load_intent_classifier


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]
    training_config = configs["training"]

    input_path = base_config["files"]["synthetic_queries_csv"]

    intent_config = training_config["intent_classifier"]
    test_size = training_config["training"]["test_size"]
    random_seed = training_config["training"]["random_seed"]

    model_path = intent_config["model_output_path"]

    queries_df = pd.read_csv(input_path)
    dataset = prepare_intent_dataset(queries_df)
    split = split_intent_dataset(
        dataset,
        test_size=test_size,
        random_seed=random_seed,
    )

    products_df = pd.read_parquet(base_config["files"]["processed_products_parquet"])
    entity_extractor = RuleBasedEntityExtractor.from_products(products_df)
    model = load_intent_classifier(model_path)
    intent_predictor = HybridIntentPredictor(
        model=model,
        entity_extractor=entity_extractor,
        )
    predictions = intent_predictor.predict_batch(split.test_texts)
    predicted_labels = [prediction.intent for prediction in predictions]
    prediction_sources = [prediction.source for prediction in predictions]
    model_intents = [prediction.model_intent for prediction in predictions]

    result = evaluate_classifier(
        true_labels=split.test_labels,
        predicted_labels=predicted_labels,
    )

    evaluation_df = pd.DataFrame(
        {
        "query": split.test_texts,
        "true_intent": split.test_labels,
        "predicted_intent": predicted_labels,
        "prediction_source": prediction_sources,
        "model_intent": model_intents,
        }
    )
    evaluation_df["is_correct"] = (
        evaluation_df["true_intent"] == evaluation_df["predicted_intent"]
    )

    print("Intent classifier evaluation")
    print(f"Dataset: {input_path}")
    print(f"Model: {model_path}")
    print(f"Test examples: {len(split.test_texts)}")
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"Macro F1: {result.macro_f1:.2%}")
    print(f"Weighted F1: {result.weighted_f1:.2%}")
    print()
    print("Classification report:")
    print(result.classification_report_text)
    print("Confusion matrix:")
    print(result.confusion_matrix.to_string())
    print()
    print("Example predictions:")
    print(evaluation_df.head(20).to_string(index=False))

    mistakes_df = evaluation_df[~evaluation_df["is_correct"]]

    print()
    print("Mistakes:")
    if mistakes_df.empty:
        print("No mistakes found.")
    else:
        print(mistakes_df.to_string(index=False))


if __name__ == "__main__":
    main()