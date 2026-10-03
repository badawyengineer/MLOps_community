# Monitoring runbook

## What is exported (`GET /metrics`)
| Series | Meaning |
|---|---|
| `prodml_requests_total{path,status}`, `prodml_request_latency_seconds` | traffic and latency per route |
| `prodml_abstentions_total`, `prodml_top_score` | refusals and the retrieval-confidence distribution |
| `prodml_query_centroid_cosine`, `prodml_query_drift_drop` | query-embedding drift vs the evaluation questions |
| `prodml_llm_tokens_total{kind}`, `prodml_llm_cost_usd_total` | token usage and estimated spend (set `PRODML_LLM_PRICE_IN_PER_1K` / `_OUT_`) |
| `prodml_guardrail_flags_total{flag}`, `prodml_llm_fallbacks_total` | PII redactions, hallucinated citations, LLM outages |
| `prodml_faithfulness_proxy` | online lexical faithfulness (NOT RAGAS) |
| `prodml_eval_faithfulness{source}` | pushed by `prodml-monitor`; `source` is `proxy` or `ragas` |

## Start the stack
```bash
docker compose -f docker/docker-compose.yml --profile observability up --build
# Grafana http://localhost:3001 (admin/admin) -> dashboard "Legal QA - RAG health"
# Prometheus http://localhost:9090/alerts shows the alert rules
```

## Evaluate quality on a schedule and push it to Prometheus
```bash
prodml-monitor --push http://localhost:9091                     # lexical proxy, free, offline
prodml-monitor --ragas --push http://localhost:9091             # real RAGAS: needs `pip install -e ".[ragas]"` + a judge-LLM key (costs money)
```
Run it from cron / a scheduled GitHub Action. The `FaithfulnessLow` alert fires below 0.80.

## Langfuse (per-request traces) - self-hosted
1. Self-host with Langfuse's official compose file (it needs Postgres, ClickHouse, Redis and MinIO, so it is not bundled here):
   `git clone https://github.com/langfuse/langfuse && cd langfuse && docker compose up -d`  -> http://localhost:3000
2. Create a project in the UI and copy its public/secret keys.
3. Start the API with:
```bash
export PRODML_LANGFUSE_HOST=http://localhost:3000
export PRODML_LANGFUSE_PUBLIC_KEY=pk-lf-...
export PRODML_LANGFUSE_SECRET_KEY=sk-lf-...
```
Every `/ask` then produces a trace with a `retrieve` span, a `generation` (with token usage) and a
`faithfulness_proxy` score. Tracing is a no-op without the keys and can never fail a request.
