from __future__ import annotations

import pandas as pd

from querysense.config import load_project_configs
from querysense.data.intent_dataset import prepare_intent_dataset, split_intent_dataset
from querysense.training.train_intent import save_intent_classifier, train_intent_classifier


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]
    training_config = configs["training"]

    input_path = base_config["files"]["synthetic_queries_csv"]

    intent_config = training_config["intent_classifier"]
    test_size = training_config["training"]["test_size"]
    random_seed = training_config["training"]["random_seed"]

    max_features = intent_config["max_features"]
    ngram_range = tuple(intent_config["ngram_range"])
    model_output_path = intent_config["model_output_path"]
    label_output_path = intent_config["label_output_path"]

    queries_df = pd.read_csv(input_path)
    dataset = prepare_intent_dataset(queries_df)
    split = split_intent_dataset(
        dataset,
        test_size=test_size,
        random_seed=random_seed,
    )

    trained_model = train_intent_classifier(
        split,
        max_features=max_features,
        ngram_range=ngram_range,
        random_seed=random_seed,
    )

    save_intent_classifier(
        trained_model,
        model_output_path=model_output_path,
        label_output_path=label_output_path,
    )

    print("Intent classifier training complete")
    print(f"Training examples: {len(split.train_texts)}")
    print(f"Test examples: {len(split.test_texts)}")
    print(f"Labels: {trained_model.labels}")
    print(f"Saved model to: {model_output_path}")
    print(f"Saved labels to: {label_output_path}")


if __name__ == "__main__":
    main()