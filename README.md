# Arabic Legal Q&A - Egyptian Civil Code (MLOps Practitioner, Project 2: LLM / RAG)

Ask a question in Arabic or English; get an answer **cited to Civil Code article numbers**, with a
confidence score - or a refusal when the corpus doesn't cover the question.

> **Status:** 16 of the 34 handbook checklist items are done, 11 partial, 7 not done (all 7 need a
> GPU, a judge-LLM key, or the real PDF). See **[reports/final-checklist.md](reports/final-checklist.md)**
> for every item, its evidence, and exactly what is missing. Quality numbers below come from a
> **5-article sample corpus and a 23-question bootstrap eval set**: they prove the machinery works,
> not that the answers are good.

## Run it - 3 commands
```bash
git clone <this-repo> && cd mlops-legal-qa
docker compose -f docker/docker-compose.yml up --build -d        # API on http://localhost:8000
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "Must contracts be performed in good faith?"}'
```
No Docker? `pip install -e ".[dev]"` then `uvicorn prodml.api.main:app` (then the same `curl`).

## Easiest way to test everything
```bash
pip install -e ".[dev,tracking,fixtures,pdf]"
prodml-smoke                       # 12 PASS/FAIL checks in ~2 s: every endpoint, abstention, streaming,
                                   # metrics, guardrails, corpus validation, retrieval-quality gate
prodml-smoke --url http://localhost:8000     # same checks against a running server / the Docker container
pytest                             # 151 tests, 91% coverage
```

## What's in it
| Capability | How to run / see it |
|---|---|
| Bilingual PDF -> per-article JSON | `prodml-ingest --dump 3` (inspect columns), `prodml-ingest`, `prodml-validate --strict --spot-check 20` |
| Chunk by article -> embed -> vector store | `prodml-index` (baked into the Docker image) |
| Retrieval comparison + abstention tuning | `prodml-eval --compare --sweep --k 1` |
| 12-run MLflow grid, champion in Registry | `prodml-experiments --plot reports/mlflow_runs_comparison.png`, then `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Reproducible pipeline | `dvc repro` (corpus -> validate -> index -> evaluate) |
| Real LLM (vLLM / any OpenAI-compatible API) | set `PRODML_GENERATION_BACKEND=vllm`, `PRODML_LLM_BASE_URL`, `PRODML_LLM_MODEL` ([docs/gpu-runbook.md](docs/gpu-runbook.md)) |
| Streaming | `curl -N -X POST localhost:8000/ask/stream -H 'Content-Type: application/json' -d '{"question":"..."}'` |
| BentoML service | `bentoml serve prodml.bento_service:LegalQAService`; `bentoml build` |
| Guardrails | PII redaction + hallucinated-citation check on every answer |
| Batch re-indexing | `prodml-reindex new_articles.json [--dry-run]` |
| Load test (50 users) | `locust -f loadtest/locustfile.py ...` -> [reports/loadtest/SUMMARY.md](reports/loadtest/SUMMARY.md) |
| Monitoring stack | `docker compose -f docker/docker-compose.yml --profile observability up` ([docs/monitoring.md](docs/monitoring.md)) |
| Quality monitor -> Prometheus + MLflow | `prodml-monitor --push http://localhost:9091 --mlflow` |
| LLM before/after (quantization) | `prodml-compare-llms --a-url ... --b-url ...` |

Architecture (all five sessions): **[docs/architecture.md](docs/architecture.md)**.

## Measured results (sample corpus - read the caveats)
- **Retrieval** (23-question bootstrap set, k=1, threshold 0.20, `hybrid-dense`): hit@1 0.94, MRR 0.94,
  0 false refusals, refuses 6 of 7 out-of-scope questions. The threshold was tuned on this same set, so
  the abstention figure is optimistic.
- **Experiments**: 12 runs; the offline embeddings (hashing/LSA) tie the TF-IDF+BM25 hybrid at best.
  A pure-LSA model had the best MRR but could never refuse a question, so runs are ranked on retrieval
  *and* abstention quality. A real multilingual embedding model is the expected upgrade; it has not been run.
- **Load** (50 users, extractive backend, 1 worker, shared CPU): ~46 req/s, p50 5 ms, p95 110 ms, p99 680 ms, 0 failures.
- **Faithfulness** is a lexical *proxy* (labelled `proxy` everywhere), not RAGAS; the extractive generator
  copies the article verbatim, so it is ~1.0 until a real LLM is plugged in.

## Configuration (environment variables, prefix `PRODML_`)
`RETRIEVER` (default `hybrid-dense`), `EMBEDDING_MODEL` (`hashing-1024` | `lsa-32` | `<org>/<hf-model>`), `MIN_CONFIDENCE` (0.2),
`CHUNK_CHARS` / `CHUNK_OVERLAP` (0 = one chunk per article), `GENERATION_BACKEND` (`extractive` | `vllm` | `api`),
`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_PRICE_IN_PER_1K` / `_OUT_` (for the cost panel),
`LANGFUSE_HOST` / `_PUBLIC_KEY` / `_SECRET_KEY`, `REPEALED_RANGES` (`54-80`).

## Canary rollout
`docker compose -f docker/docker-compose.yml --profile canary up` puts nginx in front of a stable and a
candidate container (90/10, [docker/canary/nginx.conf](docker/canary/nginx.conf)). Stages: **10% -> 50% -> 100%**, advancing only
while abstention rate, p95 latency, `prodml_faithfulness_proxy` and guardrail flags stay at baseline in
Grafana; **roll back** by setting the canary weight to 0. Every response carries `model_version`, so you can
see which side answered. *(Config only - not executed in the build environment.)*

## Using the real Civil Code PDF (the biggest remaining step)
1. Put it at `data/raw/egyptian_civil_code.pdf`, then `prodml-ingest --dump 3` and **read the columns**: layouts differ, and the extractor was only tested on a synthetic fixture.
2. Fix patterns in `src/prodml/ingest.py` if needed, then `prodml-ingest && prodml-validate --strict --spot-check 20`.
3. `dvc add data/raw/egyptian_civil_code.pdf`, wire the PDF into `dvc.yaml` (steps at its top), `dvc repro`.
4. Expand `data/eval/questions.json` to 50-100 items, re-run `prodml-eval --sweep`, **re-tune `PRODML_MIN_CONFIDENCE`**, update `data/eval/thresholds.json`.

## Layout
```
src/prodml/      ingest, corpus, chunking, embedders, vectorstore, retriever, rag (predictor), llm, guardrails,
                 faithfulness, tracing, drift, metrics, evaluation, experiments, monitor, reindex, smoke, bento_service, api/
data/eval/       labelled questions + CI regression thresholds        loadtest/   locustfile
docker/          Dockerfile, compose, Prometheus/Grafana/Alertmanager, canary    docs/   architecture, monitoring, gpu-runbook
reports/         final-checklist, module reports, load test, eval results         tests/  151 tests + synthetic PDF fixture
```
*Informational retrieval over statute text only; not legal advice.*
