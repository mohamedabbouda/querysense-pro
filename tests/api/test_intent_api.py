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
    assert data["entities"]["brand"] == "sony"
    assert data["entities"]["category"] == "electronics"
    assert data["entities"]["subcategory"] == "headphones"
    assert data["entities"]["color"] == "black"
    assert data["entities"]["max_price"] == 300.0
    recommended_filters = {
    filter_["name"]: filter_["value"]
    for filter_ in data["recommended_filters"]
    }
    assert recommended_filters["brand"] == "sony"
    assert recommended_filters["subcategory"] == "headphones"
    assert recommended_filters["color"] == "black"
    assert recommended_filters["max_price"] == 300.0
    
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


def test_recommend_filters() -> None:
    response = client.post(
        "/recommend-filters",
        json={"query": "sony black headphones under 300"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "sony black headphones under 300"
    assert data["normalized_query"] == "sony black headphones under 300"

    filters = {
        filter_["name"]: filter_["value"]
        for filter_ in data["filters"]
    }

    assert filters["brand"] == "sony"
    assert filters["category"] == "electronics"
    assert filters["subcategory"] == "headphones"
    assert filters["color"] == "black"
    assert filters["max_price"] == 300.0


def test_recommend_filters_rejects_empty_query() -> None:
    response = client.post(
        "/recommend-filters",
        json={"query": ""},
    )

    assert response.status_code == 422