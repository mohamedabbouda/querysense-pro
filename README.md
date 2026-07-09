# QuerySense Pro

QuerySense Pro is a production-style e-commerce query understanding and product retrieval system.

The goal is to simulate a real search-query intelligence system similar to those used in large e-commerce platforms.

The system takes noisy user search queries and performs:

* Query normalization
* Query rewriting
* Intent classification
* Entity extraction
* Category prediction
* Filter recommendation
* Lexical retrieval with BM25
* Semantic retrieval with embeddings
* Hybrid ranking
* API inference with FastAPI
* Active learning for annotation
* Evaluation and error analysis

---

## Example

Input:

```text
iphon blak
```

Output:

```json
{
  "raw_query": "iphon blak",
  "normalized_query": "iphone black",
  "tokens": ["iphon", "blak"],
  "corrected_tokens": ["iphone", "black"],
  "corrections": {
    "iphon": "iphone",
    "blak": "black"
  }
}
```

---

## Current Project Status

### Completed

* Professional Python project structure
* Config-driven setup
* Product schema validation with Pydantic
* Sample e-commerce product catalog
* Processed product catalog in Parquet format
* Synthetic query generation
* Query normalization
* Typo correction and synonym normalization
* Noisy query benchmark
* Unit tests
* Ruff quality checks

### In Progress

* Query understanding pipeline

### Next Milestones

* Entity extraction
* Intent classification
* Category prediction
* Filter recommendation
* BM25 retrieval
* Semantic retrieval
* Hybrid ranking
* FastAPI deployment
* Active learning annotation UI

---

## Data Pipeline

```text
data/samples/sample_products.csv
        ↓
schema validation
        ↓
data/processed/products.parquet
        ↓
synthetic query generation
        ↓
data/processed/synthetic_queries.csv
```

---

## Query Normalization Pipeline

```text
raw query
        ↓
basic text normalization
        ↓
tokenization
        ↓
typo and synonym mapping
        ↓
normalized query + corrections
```

---

## Run the Project

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Prepare product data:

```bash
python scripts/prepare_data.py
```

Generate synthetic queries:

```bash
python scripts/generate_queries.py
```

Run normalization benchmark:

```bash
python scripts/evaluate_normalization.py
```

Run tests:

```bash
pytest
ruff check src tests scripts
```

---

## Milestone 2 Final Checks

Run:

```bash
python scripts/evaluate_normalization.py
pytest
ruff check src tests scripts
```

Expected result:

```text
25 tests passing
Ruff checks passing
```

---

## Current Quality Checks

* 25 tests passing
* Ruff checks passing

---

## Project Structure

```text
src/querysense/data
```

Data loading, validation, and query dataset creation.

```text
src/querysense/query_understanding
```

Query normalization and future NLP modules.

```text
src/querysense/retrieval
```

Search, ranking, and reranking.

```text
src/querysense/evaluation
```

Metrics and error analysis.

```text
src/querysense/api
```

FastAPI service.

```text
src/querysense/active_learning
```

Feedback and annotation logic.

```text
src/querysense/monitoring
```

Logging and monitoring.
