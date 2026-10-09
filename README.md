# Arabic Legal Q&A - Egyptian Civil Code (MLOps Practitioner, Project 2: LLM / RAG)

[![CI/CD](https://github.com/badawyengineer/MLOps_community/actions/workflows/ci.yml/badge.svg)](https://github.com/badawyengineer/MLOps_community/actions/workflows/ci.yml)

Ask a question in Arabic or English; get an answer **cited to Civil Code article numbers**, with a
confidence score - or a refusal when the corpus doesn't cover the question.

> **Status:** 19 of the 34 handbook checklist items are done, 12 partial, 3 not done (all 3 need the
> real PDF or a judge-LLM key). The GPU part (vLLM, AWQ, load test with the LLM in the loop) has been run on Kaggle: see [GPU run](#gpu-run-vllm-serving-awq-quantization-load-test). See **[reports/final-checklist.md](reports/final-checklist.md)**
> for every item, its evidence, and exactly what is missing. Quality numbers below come from a
> **5-article sample corpus and a 23-question bootstrap eval set**: they prove the machinery works,
> not that the answers are good.

## Run it - 3 commands
```bash
git clone <this-repo> && cd mlops-legal-qa
docker compose -f docker/docker-compose.yml up --build -d        # API on http://localhost:8000
prodml-ask "Must contracts be performed in good faith?" --url http://localhost:8000      # works on Windows, macOS, Linux
```
No Docker? `pip install -e ".[dev]"` then `prodml-ask "Must contracts be performed in good faith?"` (answers in-process; no server needed),
or start the API with `uvicorn prodml.api.main:app` and use the `--url` form above.

**Windows (Command Prompt):** create and activate the environment with
`python -m venv .venv` then `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`). Use `prodml-ask` instead of `curl`: Command Prompt does not understand the single-quoted JSON in `curl` examples. (If you prefer curl there, use `curl.exe` with escaped double quotes: `-d "{\"question\": \"...\"}"`.)

## Easiest way to test everything
```bash
pip install -e ".[dev,tracking,fixtures,pdf]"
prodml-smoke                       # 12 PASS/FAIL checks in ~2 s: every endpoint, abstention, streaming,
                                   # metrics, guardrails, corpus validation, retrieval-quality gate
prodml-smoke --url http://localhost:8000     # same checks against a running server / the Docker container
pytest                             # 207 tests, 90% coverage (also run under a non-UTF-8 locale to simulate Windows)
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
| LLM-assisted eval-question drafting | `prodml-evalgen generate ...`, review by hand, `prodml-evalgen merge ...` |
| Alert notifications (local) | `prodml-alert-sink` then `prodml-alert-sink --fire-test-alert http://localhost:9093` |
| GPU session (vLLM + AWQ + Locust on the real LLM) | `notebooks/kaggle_gpu_run.ipynb` on Kaggle (executed on 2x T4; free **dry-run mode** with a fake LLM also available). Results: [GPU run](#gpu-run-vllm-serving-awq-quantization-load-test) |
| Evidence screenshots (MLflow, Grafana, Langfuse, alert) | [docs/evidence-runbook.md](docs/evidence-runbook.md) |

Architecture (all five sessions): **[docs/architecture.md](docs/architecture.md)**.

## Measured results (sample corpus - read the caveats)
- **Retrieval** (23-question bootstrap set, k=1, threshold 0.20, `hybrid-dense`): hit@1 0.94, MRR 0.94,
  0 false refusals, refuses 6 of 7 out-of-scope questions. The threshold was tuned on this same set, so
  the abstention figure is optimistic.
- **Bilingual consistency**: the same question in Arabic and English reaches the same decision in 4 of 4 matched pairs (a CI gate allows at most one disagreement). Four pairs is a small sample.
- **Experiments**: 12 runs; the offline embeddings (hashing/LSA) tie the TF-IDF+BM25 hybrid at best.
  A pure-LSA model had the best MRR but could never refuse a question, so runs are ranked on retrieval
  *and* abstention quality. A real multilingual embedding model is the expected upgrade; it has not been run.
- **Load, extractive backend** (50 users, 1 worker, shared CPU): ~46 req/s, p50 5 ms, p95 110 ms, p99 680 ms, 0 failures (archived in `reports/loadtest/SUMMARY_extractive.md`). **Load with the real LLM in the loop** is in the GPU run section below.
- **Faithfulness** is a lexical *proxy* (labelled `proxy` everywhere), not RAGAS; the extractive generator
  copies the article verbatim, so it is ~1.0. With the real LLM the proxy is about 0.5 (see the GPU run section).

## Running the GPU part on Kaggle (vLLM + AWQ + real-LLM load test)
1. Upload `mlops-legal-qa-v5.tar.gz` as a Kaggle **Dataset** (or attach a `GH_TOKEN` secret for the private repo).
2. Import `notebooks/kaggle_gpu_run.ipynb`, attach the Dataset with *Add Input*, set *Internet = On*.
3. **Dry run first** (free CPU session, `DRY_RUN = True`): a fake LLM (`prodml-mock-llm`) proves the notebook works without spending GPU quota.
4. Real run: *Accelerator = GPU T4 x2* (**not P100**), `DRY_RUN = False`, Run All. Outputs: `optimization.md`, `loadtest/`, `answers.json`, `run_info.json`, zipped.

Dry-run numbers describe a stub and are never results.

## GPU run: vLLM serving, AWQ quantization, load test

Measured on Kaggle (2x Tesla T4, one GPU per server), vLLM 0.31.0. Notebook: `notebooks/kaggle_gpu_run.ipynb`.
Raw evidence: `reports/run_info.json`, `reports/optimization.md`, `reports/answers.json`, `reports/loadtest/`, `reports/logs/`.

- **Model served:** `Qwen/Qwen2.5-1.5B-Instruct` (baseline)
- **Quantized model:** `Qwen/Qwen2.5-1.5B-Instruct-AWQ` - the **publisher's official AWQ checkpoint; we did not quantize it ourselves**

### Baseline vs AWQ (16 answerable questions, retrieval shared)

| metric | baseline | AWQ-4bit |
|---|---|---|
| latency p50 (ms) | 781 | 190 |
| latency p95 (ms) | 2372 | 3292 |
| tokens / s | 48 | 94 |
| faithfulness (lexical proxy) | 0.52 | 0.38 |

- AWQ roughly halves the per-token cost and cuts median latency about 4x, but **p95 is worse** in this run.
- The faithfulness drop (0.143) **exceeds the 0.03 budget** on the proxy. It is a lexical proxy over only 16 questions, not RAGAS, so it is
  indicative only; RAGAS was not run.
- Run-to-run noise is large: an earlier run of the same comparison gave p50 628 -> 181 ms and 61 -> 99 tokens/s.
- Quality caveat: at 1.5B the model is weak in Arabic. In the 16 saved baseline answers, one contains stray Russian text, and an Arabic smoke-test answer scored faithfulness 0.0.

### Load test: 50 concurrent users, vLLM in the loop

1,934 requests in 90 s, **0 failures, 0 fallbacks to extractive text** (checked from the API's own metrics), about 21.7 req/s,
median 880 ms, p95 about 4.1 s. English `/ask` p95 is about 1.1 s; Arabic `/ask` has a long tail (p95 about 4.1 s). Details: [reports/loadtest/SUMMARY.md](reports/loadtest/SUMMARY.md).

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
reports/         final-checklist, module reports, load test, eval results         tests/  163 tests + synthetic PDF fixture
```
*Informational retrieval over statute text only; not legal advice.*
