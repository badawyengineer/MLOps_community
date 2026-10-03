# Final checklist - Project 2 (LLM / RAG, Arabic Legal Q&A)

Checked item-by-item against the *Completion checklists* in the MLOps Practitioner handbook.
**34 items: 16 done, 11 partial, 7 not done.** "Partial" always says what is missing.
Every "done" is backed by a test, a command you can run, or a file in this repo.

**Verify the done items yourself:** `pip install -e ".[dev,tracking,fixtures,pdf]" && prodml-smoke && pytest`

## Code, packaging, tracking, versioning (8 done / 6 partial / 2 not done)
| Status | Handbook item | Evidence / what is missing |
|---|---|---|
| ✅ done | src/ layout with pyproject.toml; `pip install -e .` works | pyproject.toml, src/prodml; CI installs it |
| ❌ not done | Civil Code PDF -> structured JSON, one record per article, committed | Extractor built and tested on a SYNTHETIC two-column fixture only. The committed JSON is the 5-article sample. Needs the real PDF: `prodml-ingest --dump 3`, then `prodml-ingest` |
| ⚠️ partial | Extraction validated: ints, 20-article spot check, repealed flagged | All three mechanisms exist (`validate_corpus --strict`, `--spot-check 20`, repealed flagging + `PRODML_REPEALED_RANGES`) but have only run on the sample/fixture |
| ✅ done | Ingestion pipeline: JSON -> chunk by article -> embed -> vector store | `prodml-index`; default embedder is offline hashing. A real multilingual model is wired (`PRODML_EMBEDDING_MODEL=intfloat/multilingual-e5-small`) but was never run here (no model download) |
| ✅ done | FastAPI /ask {question} -> {answer, sources[]}, sources are article citations | tests/test_api.py; sources look like `Egyptian Civil Code, Article 148` |
| ✅ done | Pydantic rejects an empty question: 422, tested | tests/test_api.py::test_ask_rejects_empty_question |
| ✅ done | /health returns {status: healthy, documents_indexed: N} | tested; returns 503 until the index is loaded |
| ⚠️ partial | Dockerfile includes vector store + embedded docs; compose up on :8000 | Written (index built at image build). `docker build` itself NEVER RAN (no daemon here), but every Dockerfile step was replayed from a fresh clone with a non-editable install: that caught and fixed a real bug (the corpus JSON was gitignored, so `COPY data/processed/` would have failed). The CI `docker` job is the first true test |
| ✅ done | README: 3 commands to run Q&A on any machine | README quick start |
| ⚠️ partial | MLflow logs chunk_size, overlap, embedding_model, faithfulness | Logged, but the faithfulness metric is the lexical PROXY (`faithfulness_proxy`), not RAGAS |
| ⚠️ partial | >=5 runs compared; MLflow screenshot in /reports/ | 12 runs logged + a chart generated from them (reports/mlflow_runs_comparison.png). The chart is NOT a UI screenshot: take one yourself |
| ✅ done | Best chunking config registered in the MLflow Registry | `legal-qa-retriever@champion`; loaded back and predicted in a test |
| ❌ not done | PDF and JSON both DVC-tracked; `dvc pull` fetches, `dvc repro` rebuilds JSON from PDF | JSON->index is a verified DVC pipeline. PDF tracking + a DVC remote need the PDF; exact 5 steps are at the top of dvc.yaml |
| ✅ done | `dvc repro` re-indexes the corpus reproducibly | verified: 4 stages run, second run skips |
| ⚠️ partial | GitHub Actions: lint -> test -> rebuild index -> push Docker image | Written and YAML-validated (index rebuilt during image build; publish job needs DOCKERHUB_* secrets). NEVER RUN on GitHub yet |
| ⚠️ partial | CI fails if RAGAS faithfulness < 0.75 on a 20-question set | Gate exists but uses the lexical proxy over the 16 answerable eval questions. A real RAGAS gate needs a judge-LLM key and a 20-question set |

## Serving, monitoring, optimization (8 done / 5 partial / 5 not done)
| Status | Handbook item | Evidence / what is missing |
|---|---|---|
| ✅ done | BentoML wraps the RAG pipeline with an async /ask | `bentoml serve prodml.bento_service:LegalQAService` verified live; `bentoml build` verified; `containerize` not run (no Docker) |
| ❌ not done | vLLM serves the generative LLM; model name in README | Needs a GPU. The OpenAI-compatible client, config and docs/gpu-runbook.md are ready and tested against mocks |
| ✅ done | Streaming: tokens appear progressively in curl output | `/ask/stream` (FastAPI) and `ask_stream` (BentoML) verified; `curl -N` |
| ✅ done | Locust report at 50 concurrent users in /reports/ | reports/loadtest/ask.html. Measured on the extractive backend (no LLM in the loop) |
| ✅ done | Batch re-indexing script tested with >=1 new document | `prodml-reindex`; tested: new article becomes retrievable, bad input writes nothing |
| ✅ done | Canary rollout config documented in README | docker/canary/nginx.conf (90/10) + README section. Config only: never executed |
| ❌ not done | RAGAS on >=50 questions, all 4 metrics logged | Needs a judge LLM + a 50-question set (have 23). `prodml-monitor --ragas` is implemented but unrun |
| ⚠️ partial | RAGAS results stored in MLflow; trend visible across sessions | `prodml-monitor --mlflow` logs every pass (tested, 2 runs). Values are proxy scores until RAGAS is run |
| ⚠️ partial | Alert: faithfulness < 0.80 -> notification triggered | Rule + Alertmanager written; the webhook URL is a placeholder you must set. Never fired |
| ⚠️ partial | Langfuse self-hosted; every /ask creates a trace with spans | Exporter implemented and tested against a mock Langfuse API. Not run against a real server (needs Docker; steps in docs/monitoring.md) |
| ⚠️ partial | RAGAS faithfulness score attached to each Langfuse trace | The proxy score is attached to every trace; per-request RAGAS is not done |
| ✅ done | Cosine similarity on query embeddings vs baseline; drift logged | `prodml_query_drift_drop` + histogram; tested that off-distribution queries raise it |
| ⚠️ partial | Token usage in Prometheus; Grafana shows cost/hour | Counters + Grafana panel defined and cross-checked against real metric names; dashboard never opened in a running Grafana |
| ✅ done | Guardrails PII detection on all /ask responses | Own regex implementation (emails, phones, national ID, cards) + hallucinated-citation check; applied to /ask, /ask/batch, /ask/stream. Not the 'Guardrails AI' library |
| ❌ not done | Generative model quantized to AWQ-4bit; weights documented | Needs a GPU. Script + steps in docs/gpu-runbook.md |
| ❌ not done | RAGAS original vs quantized: <= 0.03 faithfulness drop | Comparison harness built and tested (`prodml-compare-llms`); needs two GPU endpoints |
| ❌ not done | Latency before vs after quantization in README | Same dependency on GPU endpoints. No numbers are claimed |
| ✅ done | Final architecture diagram covering all 5 sessions | docs/architecture.md (Mermaid) and README |

## The five things that unlock most of what is left
1. **The real PDF** -> `prodml-ingest --dump 3` (read it!), `prodml-ingest`, `prodml-validate --strict`, `dvc add` the PDF. Closes 3 items and makes every metric in this repo meaningful.
2. **A judge-LLM key** -> `prodml-monitor --ragas --mlflow --push ...` on a 50-question set. Closes the RAGAS items.
3. **A GPU session (Kaggle/Colab T4)** -> docs/gpu-runbook.md. Closes vLLM, AWQ, quantization drop and latency items.
4. **Docker + GitHub** -> push the branch; the CI `docker` job builds the image, checks the baked index and non-root user, and smoke-tests it over HTTP. Add the DOCKERHUB_* secrets for the publish job.
5. **Screenshots** (MLflow UI, Grafana, Langfuse) for /reports/ - nothing here can take them for you.

## Honesty clause
No number in this repository was invented. Every figure is from a command that was run, on the 5-article
sample corpus and a 23-question bootstrap eval set - **treat all quality numbers as mechanism checks,
not results**. The faithfulness figures are a lexical proxy and are labelled `proxy` everywhere; they
are never reported as RAGAS.
