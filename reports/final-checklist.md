# Final checklist - Project 2 (LLM / RAG, Arabic Legal Q&A)

Checked item-by-item against the *Completion checklists* in the MLOps Practitioner handbook.
**34 items: 19 done, 12 partial, 3 not done.** "Partial" always says what is missing.
Every "done" is backed by a test, a command you can run, or a file in this repo.

**Verify the done items yourself:** `pip install -e ".[dev,tracking,fixtures,pdf]" && prodml-smoke && pytest`

## Code, packaging, tracking, versioning (9 done / 5 partial / 2 not done)
| Status | Handbook item | Evidence / what is missing |
|---|---|---|
| ✅ done | src/ layout with pyproject.toml; `pip install -e .` works | pyproject.toml, src/prodml; CI installs it |
| ❌ not done | Civil Code PDF -> structured JSON, one record per article, committed | Extractor built and tested on a SYNTHETIC two-column fixture only. The committed JSON is the 5-article sample. Needs the real PDF: `prodml-ingest --dump 3`, then `prodml-ingest` |
| ⚠️ partial | Extraction validated: ints, 20-article spot check, repealed flagged | All three mechanisms exist (`validate_corpus --strict`, `--spot-check 20`, repealed flagging + `PRODML_REPEALED_RANGES`) but have only run on the sample/fixture |
| ✅ done | Ingestion pipeline: JSON -> chunk by article -> embed -> vector store | `prodml-index`; default embedder is offline hashing. A real multilingual model is wired (`PRODML_EMBEDDING_MODEL=intfloat/multilingual-e5-small`) but was never run here (no model download) |
| ✅ done | FastAPI /ask {question} -> {answer, sources[]}, sources are article citations | tests/test_api.py; sources look like `Egyptian Civil Code, Article 148` |
| ✅ done | Pydantic rejects an empty question: 422, tested | tests/test_api.py::test_ask_rejects_empty_question |
| ✅ done | /health returns {status: healthy, documents_indexed: N} | tested; returns 503 until the index is loaded |
| ✅ done | Dockerfile includes vector store + embedded docs; compose up on :8000 | GitHub Actions run on PR #2 was green, which includes the `docker` job: it builds the image, checks the baked index files exist, checks the non-root user, and smoke-tests the running container over HTTP. (`docker compose up` itself was not run; compose builds the same Dockerfile.) |
| ✅ done | README: 3 commands to run Q&A on any machine | README quick start |
| ⚠️ partial | MLflow logs chunk_size, overlap, embedding_model, faithfulness | Logged, but the faithfulness metric is the lexical PROXY (`faithfulness_proxy`), not RAGAS |
| ⚠️ partial | >=5 runs compared; MLflow screenshot in /reports/ | 12 runs logged + a chart generated from them (reports/mlflow_runs_comparison.png). The chart is NOT a UI screenshot: take one yourself |
| ✅ done | Best chunking config registered in the MLflow Registry | `legal-qa-retriever@champion`; loaded back and predicted in a test |
| ❌ not done | PDF and JSON both DVC-tracked; `dvc pull` fetches, `dvc repro` rebuilds JSON from PDF | JSON->index is a verified DVC pipeline. PDF tracking + a DVC remote need the PDF; exact 5 steps are at the top of dvc.yaml |
| ✅ done | `dvc repro` re-indexes the corpus reproducibly | verified: 4 stages run, second run skips |
| ⚠️ partial | GitHub Actions: lint -> test -> rebuild index -> push Docker image | CI ran green on PR #2 (lint, tests, retrieval + faithfulness gates, smoke, Docker build/run). NOT yet verified: the Docker Hub `publish` job (needs DOCKERHUB_USERNAME / DOCKERHUB_TOKEN secrets; it now skips cleanly without them), and the first run on `main` (#2) failed in 15 s - cause unknown until its log is read |
| ⚠️ partial | CI fails if RAGAS faithfulness < 0.75 on a 20-question set | Gate exists but uses the lexical proxy over the 16 answerable eval questions. A real RAGAS gate needs a judge-LLM key and a 20-question set |

## Serving, monitoring, optimization (10 done / 7 partial / 1 not done)
| Status | Handbook item | Evidence / what is missing |
|---|---|---|
| ✅ done | BentoML wraps the RAG pipeline with an async /ask | `bentoml serve prodml.bento_service:LegalQAService` verified live; `bentoml build` verified; `containerize` not run (no Docker) |
| ✅ done | vLLM serves the generative LLM; model name in README | Ran on Kaggle (2x Tesla T4): vLLM 0.31.0 served `Qwen/Qwen2.5-1.5B-Instruct`, and the service answered through it (`prodml-ask`, `prodml-smoke` 12/12). Evidence: `reports/run_info.json`, `reports/optimization.md`, `notebooks/kaggle_gpu_run.ipynb`. Model name and results are in the README section 'GPU run' |
| ✅ done | Streaming: tokens appear progressively in curl output | `/ask/stream` (FastAPI) and `ask_stream` (BentoML) verified; `curl -N` |
| ✅ done | Locust report at 50 concurrent users in /reports/ | reports/loadtest/ask.html + SUMMARY.md: 50 users, 90 s, **vLLM in the loop**, 1,934 requests, 0 failures, 0 LLM fallbacks (checked from the API's own metrics). The earlier extractive-backend test is archived in `loadtest/SUMMARY_extractive.md` |
| ✅ done | Batch re-indexing script tested with >=1 new document | `prodml-reindex`; tested: new article becomes retrievable, bad input writes nothing |
| ✅ done | Canary rollout config documented in README | docker/canary/nginx.conf (90/10) + README section. Config only: never executed |
| ❌ not done | RAGAS on >=50 questions, all 4 metrics logged | Needs a judge LLM + 50 reviewed questions on the real corpus (have 23). `prodml-evalgen` drafts candidates with an LLM for human review; `prodml-monitor --ragas` is implemented but unrun |
| ⚠️ partial | RAGAS results stored in MLflow; trend visible across sessions | `prodml-monitor --mlflow` logs every pass (tested, 2 runs). Values are proxy scores until RAGAS is run |
| ⚠️ partial | Alert: faithfulness < 0.80 -> notification triggered | Rule + Alertmanager written. `prodml-alert-sink` (a local receiver) and `--fire-test-alert` are built and live-tested against an Alertmanager-shaped payload; the real Alertmanager -> sink path never ran (needs the Docker stack) |
| ⚠️ partial | Langfuse self-hosted; every /ask creates a trace with spans | Exporter implemented and tested against a mock Langfuse API. Not run against a real server (needs Docker; steps in docs/monitoring.md) |
| ⚠️ partial | RAGAS faithfulness score attached to each Langfuse trace | The proxy score is attached to every trace; per-request RAGAS is not done |
| ✅ done | Cosine similarity on query embeddings vs baseline; drift logged | `prodml_query_drift_drop` + histogram; tested that off-distribution queries raise it |
| ⚠️ partial | Token usage in Prometheus; Grafana shows cost/hour | Counters + Grafana panel defined and cross-checked against real metric names; dashboard never opened in a running Grafana |
| ✅ done | Guardrails PII detection on all /ask responses | Own regex implementation (emails, phones, national ID, cards) + hallucinated-citation check; applied to /ask, /ask/batch, /ask/stream. Not the 'Guardrails AI' library |
| ⚠️ partial | Generative model quantized to AWQ-4bit; weights documented | Served and measured: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`, the **publisher's official AWQ checkpoint - NOT quantized by us** (recorded in `reports/run_info.json`). Own quantization with AutoAWQ (`QUANT_OPTION = "own"`) was not run |
| ⚠️ partial | RAGAS original vs quantized: <= 0.03 faithfulness drop | Comparison ran on real endpoints (`reports/optimization.md`, n=16) but with the lexical PROXY, not RAGAS: baseline 0.52 vs AWQ 0.38, drop 0.143 - **the 0.03 budget is NOT met** on this proxy. Real RAGAS (`RUN_RAGAS = True`, judge-LLM key) was not run, so the criterion is not verified either way |
| ✅ done | Latency before vs after quantization in README | Measured, 2x T4, n=16: p50 781 -> 190 ms, tokens/s 48 -> 94, but **p95 got worse** 2372 -> 3292 ms. An earlier run of the same comparison gave p50 628 -> 181 ms, so treat the figures as noisy. Table in the README section 'GPU run' and `reports/optimization.md` |
| ✅ done | Final architecture diagram covering all 5 sessions | docs/architecture.md (Mermaid) and README |

## The five things that unlock most of what is left
1. **The real PDF** -> `prodml-ingest --dump 3` (read it!), `prodml-ingest`, `prodml-validate --strict`, `dvc add` the PDF. Closes 3 items and makes every metric in this repo meaningful.
2. **A judge-LLM key** -> `prodml-monitor --ragas --mlflow --push ...` on a 50-question set. Closes the RAGAS items.
3. ~~A GPU session~~ **Done** (Kaggle 2x T4, see `reports/run_info.json`). What is left from it: real RAGAS before/after (`RUN_RAGAS = True`, judge-LLM key) and, if wanted, quantizing the model yourself (`QUANT_OPTION = "own"`).
4. **Docker + GitHub** -> push the branch; the CI `docker` job builds the image, checks the baked index and non-root user, and smoke-tests it over HTTP. Add the DOCKERHUB_* secrets for the publish job.
5. **Screenshots** (MLflow UI, Grafana, Langfuse) for /reports/ - nothing here can take them for you.

## Honesty clause
No number in this repository was invented. Every figure is from a command that was run, on the 5-article
sample corpus and a 23-question bootstrap eval set - **treat all quality numbers as mechanism checks,
not results**. The faithfulness figures are a lexical proxy and are labelled `proxy` everywhere; they
are never reported as RAGAS. The GPU-run numbers (latency, tokens/s, load test) are real measurements on a 1.5B model from one
Kaggle session; the quantization faithfulness check used 16 questions and the proxy, so it is indicative only.
