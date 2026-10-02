# Changelog (by handbook session)

## Session 1 - Package
- `src/prodml` package, env-driven config, JSON logging with correlation IDs.
- FastAPI `/ask`, `/ask/batch`, `/health`, `/metadata`; Pydantic validation (422).
- Multi-stage Dockerfile, non-root user; 34 tests.

## Session 2 - Track & version
- Bilingual PDF ingestion (`prodml.ingest`): script-based column split (robust to column bleed and
  side-swaps), visual-order RTL repair, Arabic-Indic digits, heading hierarchy, repealed flagging.
  Schema renamed `ar_text` -> `text_ar` to match the handbook.
- Strict corpus validation + 20-article spot-check printer.
- Chunk-by-article, pluggable embedders, persisted vector store (npz + JSON, no pickle).
- Retrievers: TF-IDF, BM25 (own implementation), hybrid, dense+BM25 hybrid (default).
- Evaluation harness (hit@k, recall, MRR, abstention, threshold sweep) + CI regression gate.
- MLflow: 12-run chunking x embedding grid, champion registered in the Model Registry.
- DVC pipeline (corpus -> validate -> index -> evaluate). GitHub Actions CI/CD.

## Session 3 - Serve
- OpenAI-compatible LLM client (vLLM or any hosted API) with streaming and extractive fallback.
- `/ask/stream` (SSE) and a BentoML service with async streaming (`bentoml build` verified).
- Sync handlers so a slow LLM call never blocks the event loop.
- Locust 50-user report; batch re-indexing script; canary nginx config.

## Session 4 - Optimize
- `prodml-compare-llms` before/after harness (latency, tokens/s, faithfulness) + GPU runbook for AWQ.
- **Not done:** AWQ run, vLLM run, re-ranker distillation (GPU needed / out of scope).

## Session 5 - Monitor
- Prometheus: latency, abstentions, retrieval-score distribution, drift, tokens + cost, guardrails.
- Guardrails: PII redaction, hallucinated-citation check. Langfuse tracing over REST.
- Query-embedding drift. `prodml-monitor` -> Pushgateway + MLflow. Grafana dashboard, alert rules, Alertmanager.
- Architecture diagram; one-command smoke test.
