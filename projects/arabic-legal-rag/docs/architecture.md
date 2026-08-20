# Architecture — Arabic Legal Document Q&A

## Current Session 1 architecture

```text
                 +------------------+
                 |      Client      |
                 +--------+---------+
                          |
                          | HTTP
                          v
                 +------------------+
                 |     FastAPI      |
                 |                  |
                 | /health          |
                 | /predict         |
                 | /ask             |
                 | /feedback        |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Pydantic Schemas |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 |  LegalQAModel    |
                 |   Demo/Mock      |
                 +--------+---------+
                          |
                          v
                 Answer + Citation
```

Every request is also processed by structured logging middleware:

```text
Request
  |
  +--> request_id
  +--> endpoint
  +--> method
  |
  v
FastAPI handler
  |
  v
response
  |
  +--> status_code
  +--> latency_ms
```

## Final target architecture

```text
                    Arabic Legal Documents
                              |
                              v
                     Ingestion Pipeline
                              |
                              v
                  Cleaning + Arabic Normalization
                              |
                              v
                         Chunking
                              |
                              v
                    Embedding Model
                              |
                              v
                       Vector Database
                              ^
                              |
User --> /ask --> Query Embedding --> Retriever
                                      |
                                      v
                               Retrieved Chunks
                                      |
                                      v
                              Prompt Construction
                                      |
                                      v
                                  vLLM / LLM
                                      |
                                      v
                             Answer + Citations
                                      |
                 +--------------------+------------------+
                 |                    |                  |
               RAGAS               Langfuse          Guardrails
                 |                    |                  |
                 |                    |             PII / safety
                 v                    v                  |
              MLflow              Traces                v
                                      |              Response
                                      v
                                 Prometheus
                                      |
                                      v
                                   Grafana
```

## Design rule

The Session 1 API should remain stable while the internal model implementation
changes from a demo predictor to the real RAG pipeline.
