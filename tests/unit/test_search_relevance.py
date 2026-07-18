from __future__ import annotations

from pathlib import Path

import pytest

from querysense.evaluation.search_relevance import load_search_relevance_csv


def test_load_search_relevance_csv(tmp_path: Path) -> None:
    relevance_path = tmp_path / "search_relevance.csv"

    relevance_path.write_text(
        "query,relevant_product_ids\n"
        "office laptop,p009;p010\n"
        "headphones,p005\n",
        encoding="utf-8",
    )

    examples = load_search_relevance_csv(relevance_path)

    assert len(examples) == 2
    assert examples[0].query == "office laptop"
    assert examples[0].relevant_product_ids == {"p009", "p010"}
    assert examples[1].relevant_product_ids == {"p005"}


def test_load_search_relevance_csv_rejects_missing_columns(tmp_path: Path) -> None:
    relevance_path = tmp_path / "bad_relevance.csv"

    relevance_path.write_text(
        "query\n"
        "office laptop\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required relevance columns"):
        load_search_relevance_csv(relevance_path)