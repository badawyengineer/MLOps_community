# Arabic Legal Document Q&A

Production-oriented Session 1 foundation for an Arabic legal RAG system.

> **Important:** The current implementation is a Session 1 demo skeleton. It does
> not contain a real legal knowledge base, retrieval system, or legal LLM.
> The demo response must not be treated as legal advice.

## 1. Project goal

The final system will answer questions about Arabic legal documents using a
retrieval-augmented generation (RAG) pipeline and provide source-grounded
citations.

The intended final architecture is:

```text
Arabic Legal Documents
        |
        v
Document ingestion
        |
        v
Chunking + metadata
        |
        v
Arabic embeddings
        |
        v
Vector database
        |
        v
Question --> retrieval --> context
                         |
                         v
                    LLM / vLLM
                         |
                         v
                Answer + citations
                         |
             +-----------+-----------+
             |           |           |
           RAGAS      Langfuse   Guardrails
                         |
                    Prometheus
                         |
                      Grafana
```

Session 1 deliberately implements only the production application boundary:

```text
Client
  |
  v
FastAPI
  |
  +--> Pydantic validation
  |
  +--> LegalQAModel (demo placeholder)
  |
  +--> Structured JSON logging
```

## 2. Session 1 deliverables

- Python package with `src/` layout
- FastAPI `/predict`, `/ask`, `/health`, `/feedback`
- Pydantic request/response contracts
- Dockerfile
- Docker Compose
- Structured JSON logging
- Unit and API tests
- 80%+ coverage gate
- README and architecture documentation

## 3. Requirements

- Python 3.11+
- Docker Desktop / Docker Engine
- Git

## 4. Local setup

### Three-command setup

```bash
python -m venv .venv
```

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
```

```bash
pip install -e ".[dev]"
```

## 5. Run locally

```bash
uvicorn legal_rag.api:app --reload
```

Open:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

## 6. Test

```bash
pytest
```

Coverage is configured to fail below 80%:

```bash
pytest --cov=src/legal_rag --cov-fail-under=80
```

## 7. Docker

Build and start:

```bash
docker compose up --build
```

Check:

```bash
curl http://localhost:8000/health
```

Arabic request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"ما هي المسؤولية المدنية؟\",\"language\":\"ar\"}"
```

Stop:

```bash
docker compose down
```

## 8. API contract

### GET `/health`

Returns service health and version.

### POST `/predict`

Session 1 compatibility endpoint.

Request:

```json
{
  "question": "ما هو القانون المدني؟",
  "language": "ar"
}
```

### POST `/ask`

The future production RAG endpoint. In Session 1 it calls the same demo
model as `/predict`.

### POST `/feedback`

Request:

```json
{
  "request_id": "abc-123",
  "rating": 5,
  "comment": "Useful response"
}
```

## 9. Current model status

The current `LegalQAModel` is intentionally a placeholder. Its purpose is to
prove that the service boundary, validation, tests, Docker image, and logging
work before adding the expensive RAG components.

Later, the model layer will become:

```text
LegalRAGPipeline
  |
  +-- document ingestion
  +-- Arabic text normalization
  +-- chunking
  +-- embeddings
  +-- vector DB retrieval
  +-- prompt construction
  +-- LLM/vLLM generation
  +-- citations
```

## 10. Project roadmap

### Session 1
Package + API + Docker + testing + logging

### Session 2
MLflow + DVC + GitHub Actions and experiment/reproducibility automation

### Later project stages
RAG, RAGAS evaluation, BentoML/vLLM serving, Langfuse tracing,
Prometheus/Grafana monitoring, guardrails/PII detection, and AWQ-4bit
optimization.

## 11. Legal safety

This application is an engineering project and is not a substitute for a
qualified lawyer. Until the real document corpus and evaluation pipeline are
implemented, demo responses must not be presented as authoritative legal
answers.

## 12. Development principle

Each later project component should replace or extend the current component
without breaking the API contract:

```text
Session 1 mock model
        |
        v
Real RAG pipeline
        |
        v
Tracked/evaluated RAG
        |
        v
Production serving
        |
        v
Observed + optimized system
```
