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