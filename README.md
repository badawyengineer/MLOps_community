# Arabic Legal Q&A - Egyptian Civil Code (MLOps Practitioner, Project 2: LLM / RAG)

[![CI/CD](https://github.com/badawyengineer/MLOps_community/actions/workflows/ci.yml/badge.svg)](https://github.com/badawyengineer/MLOps_community/actions/workflows/ci.yml)

Ask a question in Arabic or English; get an answer **cited to Civil Code article numbers**, with a
confidence score - or a refusal when the corpus doesn't cover the question.

> **Status:** 21 of the 34 handbook checklist items are done, 12 partial, 1 not done (real RAGAS needs a
> judge-LLM key). Full list with evidence: **[reports/final-checklist.md](reports/final-checklist.md)**. The corpus is the **real
> Egyptian Civil Code (1,149 articles, from the bilingual PDF)**. Retrieval numbers are measured on a **121-question eval
> set that an AI assistant drafted from the article texts and that has not been reviewed by a human expert**; the GPU-run
> answer-quality numbers (further below) were measured earlier, on the 5-article sample. Read [Known limitations](#known-limitations).

## For evaluators: start here

**Do I need any credentials or accounts? No.** Clone, install, run, test and evaluate everything in the core project with
no API key, no cloud account and no GPU: the default setup is offline (real retrieval over the committed corpus,
extractive answers). Optional integrations (a real LLM endpoint, Langfuse, real RAGAS, real embedding models, Docker
monitoring, a GPU notebook) are listed with free-tier facts and offline alternatives in
**[docs/credentials-and-services.md](docs/credentials-and-services.md)**. Every environment variable is in
**[.env.example](.env.example)**. `prodml-doctor` tells you what is configured and what is missing.

### Requirements (what was actually verified)

| | Verified in the audit | Declared / expected |
|---|---|---|
| OS | Linux x86_64 (Ubuntu-based container) | CI runs `ubuntu-latest`. macOS and Windows are **not verified**: the code forces UTF-8 for file I/O and a test simulates a legacy Windows locale, but no Windows/macOS run was done |
| Python | 3.12.3 | `>=3.10` in `pyproject.toml`; CI matrix 3.11 and 3.12; Dockerfile 3.11. 3.10 not exercised |
| CPU / RAM / GPU | 1 vCPU, 3 GB RAM, **no GPU**: API peak RSS ~224 MB, index load 0.8 s, ~30 ms per query | A GPU is needed **only** for the optional vLLM/AWQ notebook (Kaggle T4 x2). Nothing in the core needs one |
| Disk | repo 7.5 MB, index 0.9 MB, venv with all extras 1.1 GB | ~2 GB free is plenty. The embedding extra adds a multi-hundred-MB model download |
| Network | `pip install` from PyPI (once) | Optional: `huggingface.co` (embedding models), your LLM/Langfuse endpoints, Docker Hub / GitHub for Docker and CI |
| Ports | 8000 (API) | 8001 mock/vLLM LLM, 5000 `mlflow ui`, 5001 alert sink; Docker stack: 9090 Prometheus, 9091 Pushgateway, 9093 Alertmanager, 3001 Grafana, 8080 canary |
| Docker | **not available, so not verified** here | Needed only for the container image / monitoring stack |
| Model downloads / seed data | none: the corpus and eval set are committed; the vector index is built in memory on first use (or `prodml-index`) | The source PDF is only needed to *rebuild* the corpus: `dvc pull` fetches it from the in-repo DVC remote `dvc-store/` (no credentials) |

### Quick start (offline, no credentials)

```bash
git clone https://github.com/badawyengineer/MLOps_community && cd MLOps_community
python -m venv .venv
source .venv/bin/activate                 # Windows cmd: .venv\Scripts\activate   PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"                   # add ,tracking,pdf,fixtures for MLflow/DVC/PDF tooling (installed in CI too)
cp .env.example .env                      # optional - every default already works. Windows: copy .env.example .env

prodml-doctor                             # config, corpus and optional-service check; must end "Nothing blocking."
prodml-smoke                              # 12 PASS/FAIL checks, ~8 s, no server needed
prodml-ask "Must contracts be performed in good faith?"   # answered in-process, cites Article 148
```
Run it from the repository root (all data paths are relative; override with `PRODML_*` variables).

**Serve the API** (terminal 1) and query it (terminal 2):
```bash
uvicorn prodml.api.main:app --port 8000   # docs at http://localhost:8000/docs
prodml-ask "ما تعريف عقد البيع؟" --url http://localhost:8000      # or: curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"..."}'
prodml-smoke --url http://localhost:8000  # the same 12 checks over real HTTP
```
Endpoints: `GET /health`, `GET /metadata`, `GET /metrics` (Prometheus), `POST /ask`, `POST /ask/batch`, `POST /ask/stream` (SSE), `GET /docs`.
A refusal looks like `"abstained": true` with no sources. Windows Command Prompt does not understand single-quoted JSON in `curl`; use `prodml-ask`.

### Mock LLM mode (exercises the real LLM code path, no key, no GPU)

`prodml-mock-llm` is a small OpenAI-compatible server that **is not a language model**: it quotes the first retrieved article and
prefixes every answer with `[MOCK LLM - deterministic stub, not a language model]`. Use it to see the HTTP client, streaming,
token counters, guardrails and the extractive fallback work. Never quote its latency, quality or token numbers as results.
```bash
prodml-mock-llm --port 8001 --model mock-model                     # terminal 1
export PRODML_GENERATION_BACKEND=vllm PRODML_LLM_BASE_URL=http://127.0.0.1:8001/v1 PRODML_LLM_MODEL=mock-model   # Windows: set ...
prodml-doctor --online                                             # "LLM endpoint ... reachable"
uvicorn prodml.api.main:app --port 8002                            # terminal 2; answers now start with [MOCK LLM ...]
```
Stop the mock and the API keeps answering from the extractive fallback (`prodml_llm_fallbacks_total` increments) - verified.
To use a real endpoint instead, set the same three variables (plus `PRODML_LLM_API_KEY` if the provider needs one).

### The lifecycle, in order (what exists and what does not)

| # | Stage | Command (from the repo root) | Input -> output | Extras / needs |
|---|---|---|---|---|
| 1 | Data acquisition | the corpus is committed; to rebuild: `dvc pull` then `prodml-ingest` | PDF -> `data/processed/articles.json` | `[pdf,tracking]` |
| 2 | Data validation | `prodml-validate --strict --spot-check 20` | JSON -> OK / errors | core |
| 3 | "Model" build | `prodml-index` | JSON -> `data/index/` (**retrieval index; this project trains no model**) | core |
| 4 | Experiment tracking | `prodml-experiments --plot reports/mlflow_runs_comparison.png`, `prodml-eval --mlflow`, then `mlflow ui --backend-store-uri sqlite:///mlflow.db` | corpus + eval set -> `mlflow.db`, `mlruns/` (git-ignored), model registry | `[tracking]` |
| 5 | Evaluation + CI gate | `prodml-eval --compare --sweep --retriever hybrid-dense --k 1 --threshold 0.3 --check data/eval/thresholds.json --out reports/eval/latest.json` | eval set -> metrics, **exit 1 on regression** | core |
| 6 | Serving | `uvicorn prodml.api.main:app --port 8000` or `docker compose -f docker/docker-compose.yml up --build` | index -> HTTP API | core / Docker |
| 7 | Monitoring | `GET /metrics`; `prodml-monitor --push <pushgateway>`; `docker compose ... --profile observability up` | answers -> Prometheus/Grafana/alerts | Docker for the stack |
| 8 | CI/CD | `.github/workflows/ci.yml` (lint, tests, gate, index, smoke, Docker build/verify/publish), `monitor.yml` | push -> checks | GitHub; publish needs Docker Hub secrets |
| 9 | Reproducibility | `dvc pull && dvc repro` (corpus -> validation -> index -> gated evaluation) | PDF -> reports | `[tracking,pdf]` (dvc) |

Where outputs land: corpus `data/processed/`, index `data/index/`, eval report `reports/eval/latest.json`, MLflow store
`mlflow.db` + `mlruns/`, experiment chart `reports/mlflow_runs_comparison.png`, alert log `reports/alerts.jsonl`, load-test
results `reports/loadtest/`.

### Testing

```bash
pytest                                    # 206 tests, ~89% coverage (gate: >=70%); ~6-7 min on 1 vCPU (needs ".[dev,tracking,pdf,fixtures]")
ruff check src tests && black --check src tests     # the lint step CI runs
prodml-eval --retriever hybrid-dense --k 1 --threshold 0.3 --check data/eval/thresholds.json   # exits 1 if retrieval regresses
```
Results of a clean-machine run, with exact commands and outputs: **[docs/clean-machine-validation.md](docs/clean-machine-validation.md)**.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `prodml-...: command not found` | The venv is not active, or the package is not installed: `pip install -e ".[dev]"` |
| `error: externally-managed-environment` (PEP 668) | Use a virtualenv (above) instead of the system Python |
| `Invalid prodml configuration ... PRODML_X: ...` | The message names the variable to fix; see `.env.example`. Values are never echoed |
| `FileNotFoundError: data/processed/articles.json` / doctor `[ERR] corpus` | Run from the repository root, or set `PRODML_CORPUS_PATH` |
| doctor `[ERR] LLM endpoint ... unreachable` | Start `prodml-mock-llm` / your server, or unset `PRODML_GENERATION_BACKEND` to return to the offline default |
| `dvc repro`: `missing data 'source': data/raw/egyptian_civil_code.pdf` | Run `dvc pull` first (it copies the PDF from the in-repo remote `dvc-store/`). If you skipped it, run stages individually: `dvc repro --single-item validate_corpus` |
| `dvc pull` says files are missing | `dvc-store/` was not cloned completely; run `git status` / re-clone. Needs `pip install -e ".[tracking]"` |
| Port already in use | `uvicorn ... --port 8010` or `PRODML_PORT` |
| Docker Compose ignores `.env` | Compose reads `.env` from `docker/`; pass `--env-file .env` |
| Garbled Arabic in a Windows console | Use Windows Terminal / a font with Arabic glyphs; output is UTF-8 (not verified on a real Windows machine) |

### Known limitations

- **Retrieval quality is modest on the real corpus**: hit@1 0.60, hit@3 0.83 on the 121-question set. Example: asking for the
  definition of a sale contract in Arabic (`ما تعريف عقد البيع؟`) returns Articles 466/419, not 418.
- **Abstention is weak**: `How do I bake sourdough bread?` is *answered* (confidence 0.388 > threshold 0.30), whereas
  `How do I make a chocolate cake?` and `كيف أطبخ الكشري؟` are refused. 9 of 19 out-of-scope eval questions get answered.
- The eval set is AI-drafted and unreviewed; the 0.30 threshold was tuned on it (optimistic).
- The Arabic text layer of the source PDF has OCR-style typos; Article 1022's first Arabic paragraph comes from a secondary source (`data/manual_fixes.json`).
- **No model is trained here**: "model" = retrieval index. The LLM (Qwen2.5-1.5B) is served/quantized in the optional GPU notebook only, and those numbers predate the real corpus.
- **Never run**: real RAGAS, real multilingual embeddings, a real Langfuse server, the Docker stack in this audit, the self-quantization script on a GPU.
- **Not implemented** although earlier docs mentioned it: `prodml-evalgen` (LLM-assisted eval-question drafting).
- The source PDF is not stored in Git itself: it lives in the in-repo DVC remote `dvc-store/` (3.4 MB). That is convenient for evaluation but is not shared cloud storage; switch the remote for real teams.
- macOS/Windows and Python 3.10 are unverified.

## Reference: more commands and measurements

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
| Alert notifications (local) | `prodml-alert-sink` (webhook receiver on :5001) then `prodml-alert-sink --fire-test-alert http://localhost:9093` (needs the Docker Alertmanager) |
| GPU session (vLLM + AWQ + Locust on the real LLM) | `notebooks/kaggle_gpu_run.ipynb` on Kaggle (executed on 2x T4; a free CPU **dry-run mode** uses the labelled `prodml-mock-llm` stub, which was missing from the repo until the evaluation-readiness audit and has not been re-run in the notebook since). Results: [GPU run](#gpu-run-vllm-serving-awq-quantization-load-test) |
| Evidence screenshots (MLflow, Grafana, Langfuse, alert) | [docs/evidence-runbook.md](docs/evidence-runbook.md) |

Architecture (all five sessions): **[docs/architecture.md](docs/architecture.md)**.

## Measured results (real corpus - read the caveats)
- **Corpus**: 1,149 articles from the real bilingual PDF; 56 are repealed placeholders (Articles 54-80 and 389-417, both read from the PDF's own repeal notes). **Article 1022 lost its Arabic marker and first paragraph in the source PDF**; `data/manual_fixes.json` restores it (paragraphs 2-3 are the PDF's own words moved from 1021; **paragraph 1 comes from a secondary, non-official source** - replace it with the Official Gazette text when you have it).
- **Eval set**: 121 items (102 answerable, 19 out of scope; 67 Arabic, 54 English). The first 23 are the original bootstrap items; the rest were drafted by an AI assistant from the article texts and spot-checked against them. **Review them by hand before trusting any figure.**
- **Retrieval** (shipped `hybrid-dense`, k=1, abstention threshold 0.30): hit@1 **0.598**, MRR 0.598 (hit@3 0.833); 11% of answerable questions are wrongly refused and 53% of out-of-scope questions are correctly refused. On the same set plain `hybrid` (BM25 + TF-IDF) gets hit@1 0.667 and `bm25` 0.657, so the lexical models beat the hashing-dense mix at k=1 and tie it at k=3. These numbers are far below the 0.94 measured on the 5-article sample: finding one article among 1,149 is much harder.
- **Abstention is weak.** 9 of the 19 out-of-scope questions are still answered, including near-domain ones ("penalty for drunk driving" scores 0.53). The offline hashing/LSA embeddings cannot separate them from civil-liability articles; a real multilingual embedding model is the expected fix and has not been run. The 0.30 threshold is the balanced-accuracy optimum on this same set, so it is optimistic.
- **Tried and dropped**: an English light stemmer gave no gain (hit@1 flat to -2 points), so it was not shipped.
- **Source-document caveat**: the Arabic text layer contains OCR-style typos (for example `العاقر` for `العقار`), which slightly hurt Arabic retrieval; the English column is cleaner.
- **Experiments** (12 runs, re-run on the real corpus; dense-only retrieval, so not comparable to the hybrid numbers above): best run `hashing-1024`, chunk 200, overlap 0 (registered as `legal-qa-retriever@champion`); the pure-LSA runs never abstain correctly. The shipped default is un-chunked; trying chunk 200 on the shipped `hybrid-dense` gave hit@1 0.63 (vs 0.60) but correct-abstain 0.37 (vs 0.53), a trade-off rather than a clear win, so the default was not changed.
- **Measured before the real corpus arrived** (5-article sample): bilingual consistency (4 of 4 matched pairs).
- **Load, extractive backend** (50 users, 1 worker, shared CPU, sample corpus): ~46 req/s, p50 5 ms, p95 110 ms, p99 680 ms, 0 failures (archived in `reports/loadtest/SUMMARY_extractive.md`). **Load with the real LLM in the loop** is in the GPU run section below.
- **Faithfulness** is a lexical *proxy* (labelled `proxy` everywhere), not RAGAS; the extractive generator
  copies the article verbatim, so it is ~1.0. With the real LLM the proxy is about 0.5 (see the GPU run section).

## Running the GPU part on Kaggle (vLLM + AWQ + real-LLM load test)
1. The repo is public, so the notebook can `git clone https://github.com/badawyengineer/MLOps_community` (no token needed); alternatively upload a copy as a Kaggle **Dataset**.
2. Import `notebooks/kaggle_gpu_run.ipynb`, attach the Dataset with *Add Input*, set *Internet = On*.
3. **Dry run first** (free CPU session, `DRY_RUN = True`): a fake LLM (`prodml-mock-llm`, a labelled stub) proves the notebook works without spending GPU quota.
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

## Configuration (environment variables)
Copy `.env.example` to `.env` (git-ignored); every variable is optional and documented there. Real environment variables override `.env`.
Prefix `PRODML_`: `RETRIEVER` (default `hybrid-dense`), `EMBEDDING_MODEL` (`hashing-1024` | `lsa-32` | `<org>/<hf-model>`), `MIN_CONFIDENCE` (0.3),
`CHUNK_CHARS` / `CHUNK_OVERLAP` (0 = one chunk per article), `GENERATION_BACKEND` (`extractive` | `vllm` | `api`),
`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_PRICE_IN_PER_1K` / `_OUT_` (for the cost panel),
`LANGFUSE_HOST` / `_PUBLIC_KEY` / `_SECRET_KEY` (all three or none), `REPEALED_RANGES` (`54-80,389-417`; ranges announced by the PDF itself are merged in).
Unprefixed, read from the environment: `MLFLOW_TRACKING_URI`, `OPENAI_API_KEY` (real RAGAS only). Invalid combinations stop startup with a message naming the variable.

## Canary rollout
`docker compose -f docker/docker-compose.yml --profile canary up` puts nginx in front of a stable and a
candidate container (90/10, [docker/canary/nginx.conf](docker/canary/nginx.conf)). Stages: **10% -> 50% -> 100%**, advancing only
while abstention rate, p95 latency, `prodml_faithfulness_proxy` and guardrail flags stay at baseline in
Grafana; **roll back** by setting the canary weight to 0. Every response carries `model_version`, so you can
see which side answered. *(Config only - not executed in the build environment.)*

## Rebuilding from the real Civil Code PDF
- The PDF (`data/raw/egyptian_civil_code.pdf`) is **DVC-tracked**: the pointer `data/raw/egyptian_civil_code.pdf.dvc` is in Git and the file itself sits in the default DVC remote `dvc-store/`, a directory inside this repository (see `.dvc/config`). On a fresh clone: `pip install -e ".[tracking,pdf]" && dvc pull && dvc repro`, with no account or credentials. To use cloud storage instead: `dvc remote add -d storage s3://... --force && dvc push`.
- `dvc repro` runs: ingest PDF -> strict validation -> index -> evaluation with the CI gates. To look at the raw extraction first: `prodml-ingest --dump 3` (a `*` marks bold lines, which are treated as headings).
- How the real layout is handled (`src/prodml/ingest.py`): bilingual pages are split by script; visual-order Arabic is detected and un-reversed with lam-alef ligature glyphs kept intact; bold lines are Book/Chapter/Section/topic headings; the promulgation-law preamble is skipped; repeal notes ("Articles 389-417 repealed") are read, their footnotes dropped and placeholders inserted; cross-references such as "Article 444." are not mistaken for article markers.
- **Real embeddings, GPU re-run, own quantization**: see [docs/real-embeddings-and-gpu-runbook.md](docs/real-embeddings-and-gpu-runbook.md) (needs Hugging Face access / a GPU).
- **Next for quality**: review `data/eval/questions.json` by hand, run a real multilingual embedding model (`PRODML_EMBEDDING_MODEL`), then re-run `prodml-eval --sweep`, `prodml-experiments` and update `data/eval/thresholds.json`.

## Layout
```
src/prodml/      ingest, corpus, chunking, embedders, vectorstore, retriever, rag (predictor), llm, guardrails,
                 faithfulness, tracing, drift, metrics, evaluation, experiments, monitor, reindex, smoke, bento_service, api/
data/eval/       labelled questions + CI regression thresholds        loadtest/   locustfile
docker/          Dockerfile, compose, Prometheus/Grafana/Alertmanager, canary    docs/   architecture, monitoring, gpu-runbook
reports/         final-checklist, module reports, load test, eval results         tests/  206 tests + synthetic PDF fixture
```
*Informational retrieval over statute text only; not legal advice.*
