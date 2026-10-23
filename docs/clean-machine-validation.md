# Clean-machine validation report

How this was done: a directory containing exactly the files a `git clone` would contain (tracked + untracked-but-not-ignored:
124 files, **no source PDF, no vector index, no `mlflow.db`**) and a brand-new virtualenv, installed from the project's own
`pyproject.toml`. All commands ran from the repository root with **no `.env` file and no credentials**.

Environment: Linux x86_64 container, Python 3.12.3, 1 vCPU, 3 GB RAM, no GPU, **no Docker**. macOS, Windows and Python 3.10/3.11
were not tested. Timings are from this small machine while no other job ran unless noted.

## Results

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Install | `python -m venv .venv && pip install -e ".[dev]"` | OK. Extras later: `pip install -e ".[tracking,pdf,fixtures]"` OK |
| 2 | Config + environment | `prodml-doctor` | OK, exit 0: corpus 1,149 articles (1,093 live); optional services reported SKIP |
| 3 | In-process smoke | `prodml-smoke` | **12/12 PASS in 8 s** (endpoints, EN/AR asks, abstention, validation 422, batch, SSE, metrics, guardrails, corpus, retrieval gate) |
| 4 | Index build | `prodml-index` | OK: 1,149 chunks, `meta.json` + `vectors.npz` = 0.9 MB |
| 5 | Corpus validation | `prodml-validate --strict` | `OK - 1149 articles validated` |
| 6 | Evaluation + gate | `prodml-eval --retriever hybrid-dense --k 1 --threshold 0.3 --check data/eval/thresholds.json` | exit 0; hit@1 0.60, MRR 0.60, false-abstain 0.11, correct-abstain 0.53 |
| 7 | Gate fails when it should | same with a gate file requiring `hit_at_k: 0.99` | exit 1, `GATE FAIL [hybrid-dense]: hit_at_k 0.598 < 0.99` |
| 8 | Live API | `uvicorn prodml.api.main:app --port 8000` + `curl` | `/health` 200, `/metadata` 200, `/ask` EN and AR 200 with sources, empty question 422, `/metrics` 200, `/docs` 200; JSON logs, 0 tracebacks |
| 9 | Mock LLM path | `prodml-mock-llm` + API with `PRODML_GENERATION_BACKEND=vllm` | answers carry the `[MOCK LLM ...]` label and cite Article 148; `/ask/stream` streams; token counters recorded; `prodml-doctor --online` reports the endpoint reachable |
| 10 | LLM endpoint down | stop the mock, ask again | API still answers (extractive), `prodml_llm_fallbacks_total` = 1; `prodml-doctor --online` exits 1 with `[ERR] LLM endpoint ... unreachable` |
| 11 | Invalid configuration | `PRODML_GENERATION_BACKEND=api` without a model; bad retriever/port; partial Langfuse; non-numeric port | each stops startup with one readable message naming the variable; no traceback; secret values never printed |
| 12 | Secret-leak canary | canary values in `PRODML_LLM_API_KEY`, Langfuse keys, `OPENAI_API_KEY`, then `/ask`, `/metadata`, `/metrics`, `/openapi.json`, API log, `prodml-doctor`, `repr(settings)` | **0 occurrences** in all seven outputs (one leak of the Langfuse *public* key in `repr` was found and fixed) |
| 13 | MLflow | `prodml-eval ... --mlflow`; `prodml-experiments --plot ...` | experiment + run + metrics persisted in `mlflow.db`; 12-run grid in **68 s**; `legal-qa-retriever` registered with alias `@champion`; `mlruns/` written (1.3 MB) |
| 14 | DVC (clone-equivalent repo) | `dvc repro` | **FAILS** without the PDF: `missing data 'source': data/raw/egyptian_civil_code.pdf` |
| 15 | DVC pull | `dvc pull` | **FAILS**: no remote configured |
| 16 | DVC stages individually | `dvc repro --single-item validate_corpus` / `build_index` / `evaluate` | all three exit 0 |
| 17 | Alert receiver | `prodml-alert-sink` + `curl -X POST /alerts` | HTTP 200, `[FIRING] FaithfulnessLow (critical) ...` printed, JSONL line written; `--fire-test-alert` against nothing prints a clear error, exit 1 |
| 18 | Full test suite | `pytest` (clean venv, all extras) | **206 passed, 0 failed, 0 skipped**, coverage 88.9% (gate 70%), 6 min 26 s; 8 third-party deprecation warnings |
| 19 | Lint / format | `ruff check src tests && black --check src tests` | clean |
| 20 | Secret scan | key-format regexes over the working tree and all 36 commits (`git log --all -p`) | **no matches**; no key/`.env`/`kaggle.json`-style file is tracked |
| 21 | Resource use | `predictor.load()` + one query | load 0.8 s, first query ~30 ms, peak RSS ~224 MB (measured while the test suite was running, so slightly pessimistic) |

## Not verified (and why)

| Item | Blocker | Next action |
|---|---|---|
| Docker image build, `docker compose`, Prometheus / Grafana / Alertmanager, canary | No Docker in the audit environment | Run `docker compose --env-file .env -f docker/docker-compose.yml up --build` and `prodml-smoke --url http://localhost:8000`; the author's CI job builds and smoke-tests the image |
| Alertmanager -> `prodml-alert-sink` delivery | Needs the Docker stack | `docs/evidence-runbook.md` Part B |
| GitHub Actions run on the final code | Changes are uncommitted/unpushed here | Push, then read the Actions tab (the first run on `main` failed and was never read) |
| Real LLM endpoint, real Langfuse server, real RAGAS, real embedding models | No GPU, no keys, no Hugging Face access | `docs/credentials-and-services.md`, `docs/real-embeddings-and-gpu-runbook.md` |
| GPU notebook (vLLM, AWQ, self-quantization script) | No GPU | Kaggle runbook |
| Rebuilding the corpus from the PDF in the clean copy | The PDF is not in Git and there is no DVC remote | Author: add a remote and `dvc push`, or supply the PDF |
| macOS / Windows / Python 3.10 | Not available | - |

## Defects found by this audit and fixed

1. `prodml-mock-llm`, `prodml-alert-sink` were documented everywhere but did not exist in the repository or its history -> implemented, tested.
2. `prodml-evalgen` is documented but does not exist -> README row removed, listed under limitations (not implemented).
3. `prodml-eval --compare` gated `tfidf` instead of the shipped retriever when the shipped one was not an offline baseline (CI would fail) -> fixed earlier, regression test added.
4. `.env` was not loaded for unprefixed variables (`MLFLOW_TRACKING_URI`, `OPENAI_API_KEY`) -> loaded now.
5. No startup validation of configuration; the Langfuse public key appeared in `repr(settings)` -> validated, hidden.
6. `.gitignore` did not cover `.env.*`, private keys, `kaggle.json`, `.dvc/config.local` -> added.
7. The README quick start used `prodml-ask` after a Docker-only step without installing the package -> rewritten.

## Defects found and NOT fixed (behavioural)

- Out-of-scope refusal is unreliable (`How do I bake sourdough bread?` is answered with confidence 0.388).
- Arabic `ما تعريف عقد البيع؟` retrieves Articles 466/419 instead of 418.
Both follow from the offline hashing embeddings and are quantified in the README ("Measured results"); a real multilingual embedding model is the expected fix.
