from __future__ import annotations

from fastapi.testclient import TestClient

from querysense.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_intent() -> None:
    response = client.post(
        "/predict-intent",
        json={"query": "sony black headphones"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "sony black headphones"
    assert data["normalized_query"] == "sony black headphones"
    assert data["intent"] == "product_search"
    assert data["source"] in {"rule", "model"}
    assert data["model_intent"] is not None


def test_predict_intent_rejects_empty_query() -> None:
    response = client.post(
        "/predict-intent",
        json={"query": ""},
    )

    assert response.status_code == 422



def test_search_products() -> None:
    response = client.post(
        "/search",
        json={"query": "sony black headphones under 300"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "sony black headphones under 300"
    assert data["normalized_query"] == "sony black headphones under 300"
    assert data["intent"] == "price_search"
    assert len(data["results"]) >= 1

    first_result = data["results"][0]

    assert first_result["brand"].lower() == "sony"
    assert first_result["subcategory"].lower() == "headphones"
    assert first_result["price"] <= 300
    assert first_result["score"] > 0
    assert "brand" in first_result["match_reasons"]
    assert "subcategory" in first_result["match_reasons"]
    assert "max_price" in first_result["match_reasons"]


def test_search_products_rejects_empty_query() -> None:
    response = client.post(
        "/search",
        json={"query": ""},
    )

    assert response.status_code == 422