from __future__ import annotations

import pandas as pd

from querysense.config import load_project_configs
from querysense.data.intent_dataset import prepare_intent_dataset, split_intent_dataset


def main() -> None:
    configs = load_project_configs()
    base_config = configs["base"]
    training_config = configs["training"]

    input_path = base_config["files"]["synthetic_queries_csv"]
    test_size = training_config["training"]["test_size"]
    random_seed = training_config["training"]["random_seed"]

    queries_df = pd.read_csv(input_path)
    dataset = prepare_intent_dataset(queries_df)
    split = split_intent_dataset(
        dataset,
        test_size=test_size,
        random_seed=random_seed,
    )

    print("Intent classification dataset")
    print(f"Input path: {input_path}")
    print(f"Total examples: {len(dataset)}")
    print(f"Train examples: {len(split.train_texts)}")
    print(f"Test examples: {len(split.test_texts)}")
    print()
    print("Intent distribution:")
    print(dataset["intent"].value_counts().to_string())
    print()
    print("Sample rows:")
    print(dataset.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
    