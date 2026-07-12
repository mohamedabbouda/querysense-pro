# QuerySense Pro

QuerySense Pro is a production-style NLP project for e-commerce search query understanding.

It normalizes noisy user queries, extracts structured entities, predicts search intent using a hybrid rule-based and machine learning approach, evaluates model behavior with detailed metrics, and exposes predictions through a FastAPI endpoint.

## Features

- Query normalization and typo correction
- Product catalog validation with Pydantic
- Synthetic query generation
- Rule-based entity extraction
- Entity extraction evaluation
- TF-IDF + Logistic Regression intent classification
- Hybrid rule + ML intent prediction
- Classification reports and confusion matrices
- Reusable intent prediction service
- FastAPI endpoint for intent prediction
- Unit and API tests
- Ruff linting
- Rule-based product filtering
- Product ranking with match reasons
- End-to-end product search API

## Project structure

```text
querysense-pro/
├── configs/
├── data/
│   ├── samples/
│   └── processed/
├── models/
├── scripts/
├── src/querysense/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── query_understanding/
│   └── training/
└── tests/
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data pipeline

Prepare the product catalog:

```bash
python scripts/prepare_data.py
```

Generate synthetic queries:

```bash
python scripts/generate_queries.py
```

Extract entities:

```bash
python scripts/extract_entities.py
```

Evaluate entity extraction:

```bash
python scripts/evaluate_entities.py
```

## Intent classification

Prepare the intent dataset:

```bash
python scripts/prepare_intent_dataset.py
```

Train the intent classifier:

```bash
python scripts/train_intent.py
```

Evaluate the intent classifier:

```bash
python scripts/evaluate_intent.py
```

Run example predictions:

```bash
python scripts/predict_intent.py
```
Run product search examples:

```bash
python scripts/search_products.py

## API

Start the FastAPI app:

```bash
uvicorn querysense.api.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```http
GET /health
```
Product search:

```http
POST /search
```

Example request:

```json
{
  "query": "sony black headphones under 300"
}
```

Example response:

```json
{
  "query": "sony black headphones under 300",
  "normalized_query": "sony black headphones under 300",
  "intent": "price_search",
  "results": [
    {
      "product_id": "p005",
      "title": "Sony WH-1000XM4 Wireless Headphones",
      "brand": "Sony",
      "category": "Electronics",
      "subcategory": "Headphones",
      "color": "Black",
      "size": "one-size",
      "gender": "unisex",
      "condition": "new",
      "price": 249.99,
      "currency": "EUR",
      "score": 8.0,
      "match_reasons": ["brand", "subcategory", "color", "max_price"]
    }
  ]
}
```

Intent prediction:

```http
POST /predict-intent
```

Example request:

```json
{
  "query": "sony black headphones"
}
```

Example response:

```json
{
  "query": "sony black headphones",
  "normalized_query": "sony black headphones",
  "intent": "product_search",
  "source": "rule",
  "model_intent": "product_search"
}
```

## Current benchmark results

### Entity extraction

Current entity extraction accuracy on the synthetic benchmark:

```text
Overall entity accuracy: 99.78%

brand:       100.00%
category:    100.00%
subcategory: 100.00%
color:       100.00%
size:         95.00%
gender:      100.00%
condition:   100.00%
max_price:   100.00%
```

### Intent classification

Initial TF-IDF + Logistic Regression baseline:

```text
Accuracy:    57.14%
Macro F1:    38.48%
Weighted F1: 46.53%
```

After class balancing and hybrid rule-based overrides:

```text
Accuracy:    100.00%
Macro F1:    100.00%
Weighted F1: 100.00%
```

Note: the 100% score is measured on the current synthetic benchmark. A larger and noisier dataset is needed for a more realistic production evaluation.

## Quality checks

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check src tests scripts
```

Current status:

```text
90 tests passing
Ruff checks passing
```

## Example queries

```text
sony black headphones       -> product_search
sony                        -> brand_search
headphones                  -> category_search
new sony headphones         -> filtered_product_search
sony headphones under 300   -> price_search
iphon blak                  -> product_search
```

## Tech stack

- Python
- pandas
- scikit-learn
- Pydantic
- FastAPI
- pytest
- Ruff
- joblib

## Project goal

The goal of QuerySense Pro is to demonstrate an end-to-end search query understanding pipeline similar to what is used in e-commerce search systems.

It covers:

- data preparation
- normalization
- entity extraction
- intent classification
- evaluation
- API serving
- testing
- iterative model improvement