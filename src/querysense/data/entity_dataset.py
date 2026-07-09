from __future__ import annotations

import pandas as pd

from querysense.query_understanding.entity_extractor import RuleBasedEntityExtractor


def add_extracted_entity_columns(
    queries_df: pd.DataFrame,
    extractor: RuleBasedEntityExtractor,
) -> pd.DataFrame:
    """Add extracted entity columns to a query dataset."""
    enriched_df = queries_df.copy()

    extracted_records: list[dict[str, str | float | None]] = []

    for query in enriched_df["query"]:
        entities = extractor.extract(query)
        extracted_records.append(entities.to_dict())

    extracted_df = pd.DataFrame(extracted_records)
    extracted_df = extracted_df.add_prefix("extracted_")

    return pd.concat([enriched_df, extracted_df], axis=1)