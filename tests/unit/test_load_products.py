from __future__ import annotations

import pandas as pd
import pytest

from querysense.data.load_products import ProductDataError, load_products_csv


def test_load_sample_products_successfully() -> None:
    df = load_products_csv("data/samples/sample_products.csv")

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20, 13)
    assert set(["product_id", "title", "brand", "category", "price"]).issubset(df.columns)


def test_product_price_is_numeric() -> None:
    df = load_products_csv("data/samples/sample_products.csv")

    assert pd.api.types.is_numeric_dtype(df["price"])
    assert (df["price"] >= 0).all()


def test_missing_required_column_raises_error(tmp_path) -> None:
    broken_csv = tmp_path / "broken_products.csv"

    broken_csv.write_text(
        "product_id,title,price\n"
        "P001,Example Product,10.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ProductDataError, match="Missing required columns"):
        load_products_csv(broken_csv)