# Credential and external-service inventory

Short answer for an evaluator: **no account, API key or paid service is needed to clone, install, run, test and
evaluate the core project.** The default configuration is fully offline and deterministic: real retrieval over the
committed Civil Code corpus, extractive answers, an optional *clearly labelled* mock LLM. Real credentials only enable
optional integrations, listed below.

Check what applies to your machine at any time with `prodml-doctor` (offline) or `prodml-doctor --online`
(also probes the services you configured). Variable names and defaults: [`.env.example`](../.env.example).
Free-tier statements were checked against the providers' public pages in October 2026 and can change.

## Which path should I use?

| Path | You have | What works | Setup |
|---|---|---|---|
| **0 - Offline (default)** | nothing | API, retrieval, abstention, guardrails, streaming, metrics, tests, evaluation + CI gate, MLflow (local), DVC stages that do not need the PDF | README "Quick start" |
| **B - Mock LLM** | nothing | Everything in path 0 **plus** the real LLM client code path (HTTP, streaming, token accounting, fallback) against `prodml-mock-llm`. Answers are labelled `[MOCK LLM ...]` and are **not** model output; never quote its latency/quality | README "Mock LLM mode" |
| **A - Your own keys** | free-tier accounts you create | Optional extras: Langfuse tracing (free), a real OpenAI-compatible LLM endpoint (free options vary), real RAGAS (**paid**) | Table below |
| **C - Required services** | Docker / a GPU | Only for the monitoring stack (Docker) and the GPU notebook (free Kaggle GPU). **Nothing in the core flow requires them** | `docs/evidence-runbook.md`, `docs/real-embeddings-and-gpu-runbook.md` |

## Inventory (every dependency found in the code, scripts, compose files, CI and notebook)

| Item | Required? | Purpose | Free-tier option | Offline / mock alternative |
|---|---|---|---|---|
| Python >= 3.10 and `pip install -e ".[dev]"` | **Yes** | Run everything | Free (PyPI) | - |
| `data/processed/articles.json` (committed, 1.3 MB) | **Yes** (in repo) | The Civil Code corpus: 1,149 articles | In the repository | - |
| Source PDF `data/raw/egyptian_civil_code.pdf` (2.6 MB, DVC-tracked) | No - only to *rebuild* the corpus (`prodml-ingest`, full `dvc repro`) | Corpus provenance | `dvc pull` fetches it from the **in-repo DVC remote `dvc-store/`** - no account | The committed JSON is the same data |
| `PRODML_GENERATION_BACKEND=extractive` (default) | - | Answers by quoting retrieved articles | - | This **is** the offline mode |
| LLM endpoint: `PRODML_GENERATION_BACKEND=vllm\|api`, `PRODML_LLM_BASE_URL`, `PRODML_LLM_MODEL`, `PRODML_LLM_API_KEY` (secret) | No | Generative answers via any OpenAI-compatible server (vLLM, hosted API) | Self-hosted vLLM is free but needs a GPU (free Kaggle T4 x2 ~30 h/week with phone verification, per public guides). **OpenAI's API has no free tier** (prepaid credits). Other providers: check their own free tiers | `prodml-mock-llm` (labelled stub, no key) or the extractive default |
| `PRODML_LANGFUSE_HOST` / `_PUBLIC_KEY` / `_SECRET_KEY` (all three or none) | No | LLM request tracing | **Langfuse Cloud Hobby: free, no credit card, 50k units/month, 30-day retention, 2 users** (langfuse.com/pricing). Self-hosting is free | Leave unset: tracing is skipped. Tested only against a mock server, never a real Langfuse |
| `MLFLOW_TRACKING_URI`, `PRODML_MLFLOW_ARTIFACTS` | No | Experiment tracking | MLflow is open source; the default is a local `mlflow.db` file (no server, no account) | Default local store |
| `OPENAI_API_KEY` (secret) + `pip install -e ".[ragas]"` | No | Real RAGAS faithfulness (`prodml-monitor --ragas`, optional `monitor.yml` workflow) | **None - OpenAI requires prepaid credits.** Costs real money | The built-in lexical faithfulness *proxy* (labelled `proxy` everywhere). RAGAS has never been run in this project |
| `PRODML_EMBEDDING_MODEL=<org>/<model>` + `pip install -e ".[embeddings]"` | No | Real multilingual embeddings (e.g. `intfloat/multilingual-e5-small`) | Hugging Face downloads are free for public models; needs internet to `huggingface.co`. **Never run in this project** (no network to HF in the build environment) | Default `hashing-1024` (offline, no download) |
| `HF_HOME`, `HF_TOKEN` | No | HF cache dir / gated models | Free HF account (token only for gated models) | - |
| Docker + Compose; ports 8000, 9090, 9091, 9093, 3001, 8080 | No - only for the container image and the monitoring stack | Image, Prometheus, Pushgateway, Alertmanager, Grafana, canary | Free (Docker Engine/Desktop personal use) | Run `uvicorn` directly. **The Docker build and compose stack were NOT verified in this audit (no Docker available)**; they were verified by the author's CI only |
| `GF_SECURITY_ADMIN_PASSWORD` (hard-coded `admin` in compose) | n/a | Local Grafana login | - | Local demo only; do not expose port 3001 |
| GitHub repository secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` | No | CI publishes the image on pushes to `main` | Docker Hub free personal account + access token (not re-verified here). Without them CI skips publishing with a notice | Skip |
| GitHub repository secret `OPENAI_API_KEY` | No | Optional real-RAGAS workflow | None (paid) | The workflow step is skipped when the secret is absent |
| Kaggle account + GPU notebook (`notebooks/kaggle_gpu_run.ipynb`) | No | vLLM serving, AWQ quantization, load test on a GPU | ~30 free GPU hours/week, T4 x2, phone verification (public guides; Kaggle's own page not fetched). `DRY_RUN = True` runs on CPU with the mock LLM | Dry-run mode |
| DVC remote | No | Share the PDF | Default is `dvc-store/` inside the repo (free, no credentials); S3/GDrive/etc. optional | Default |
| GitHub Actions | No | CI on the evaluator's fork | Free for public repositories | Run the same commands locally (README "Testing") |

## Credential-handling rules in this repository

- `.env` and `.env.*` are git-ignored (only `.env.example` is tracked), as are `*.pem`, `*.key`, `kaggle.json`,
  `credentials*.json` and `.dvc/config.local`.
- Secrets are `pydantic` fields with `repr=False`; `settings.safe_summary()` and `prodml-doctor` print "set/unset" only.
  A canary-secret test confirmed no canary value appears in the API log, `/ask`, `/metadata`, `/metrics`, `/openapi.json`,
  the doctor output or `repr(settings)`.
- Invalid or inconsistent configuration stops startup with a one-line message naming the variable (never the value).
- The Git history (36 commits) and the working tree were scanned for common key formats (OpenAI, GitHub, AWS, Slack,
  Hugging Face, Google, Docker, Langfuse, private-key headers): **no matches**.
- Do not paste keys into issues, commit messages or the notebook output. Supply your own through `.env` or the shell.
