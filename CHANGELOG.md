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

## Fixes found by clean-room verification
- `data/processed/articles.json` was gitignored since Module 1, so it was never committed and a fresh clone (or `COPY data/processed/` in Docker) had no corpus. Now tracked; regression test added.
- `prodml-index` silently fell back to the 5-article sample when the corpus was missing, which masked the above. It now errors unless `--allow-sample`.
- Load test: a locustfile bug (missing `catch_response=True`) crashed the stream task while stats showed 0 failures; fixed and re-run.
- Experiment ranking on MRR alone crowned a model that could never abstain; now ranks on retrieval + abstention quality.
- **Windows:** `read_text()`/`write_text()` calls without `encoding="utf-8"` crashed on Windows (default cp1252) when reading the Arabic corpus (3 test failures on a real Windows machine), and printing Arabic to a redirected console raised `UnicodeEncodeError`. All text I/O is now explicit UTF-8, every CLI forces UTF-8 output, `.gitattributes` pins LF line endings, and regression tests run the CLIs in a non-UTF-8 locale plus a static check that no text I/O omits the encoding. Added `prodml-ask` (no curl/JSON quoting).

## After the first GitHub run
- CI ran green on PR #2 (3m 12s), which also verifies the Docker image build, baked index, non-root user and HTTP smoke test.
- The merge to `main` (run #2) failed in 15 s; cause pending its log. Hardened the workflow: the Docker Hub `publish` job now skips cleanly without secrets (it previously failed at login on every push to `main`), least-privilege token permissions, a manual `workflow_dispatch` trigger.
- **Bug fixed:** `prodml-eval --compare --retriever hybrid-dense` silently gated `tfidf` instead of the shipped retriever. It now always evaluates and gates the shipped retriever; regression test added.
- Added bilingual-consistency metric (same question in Arabic and English must give the same decision) with a CI gate, `prodml-alert-sink`, `prodml-evalgen` (LLM-drafted eval questions for human review), and `notebooks/kaggle_gpu_run.ipynb`.

## Kaggle GPU notebook
- `notebooks/kaggle_gpu_run.ipynb`: standalone, uploadable to Kaggle. Finds the code (uploaded Dataset archive/folder, or git clone with a `GH_TOKEN` secret), serves baseline + AWQ with vLLM, compares them, load-tests at 50 users, zips the results. Wrong GPU (e.g. P100), missing code and read-only inputs are handled with clear messages.
- **DRY_RUN mode** with `prodml-mock-llm` (a fake OpenAI-compatible server) so the whole notebook can be rehearsed on a free CPU session. The rehearsal was executed end to end here (3 delivery variants + 2 failure paths) and found real bugs: tool lookup on PATH, `pip` pointing at a different Python, locust missing from the dependencies (now a `loadtest` extra), read-only `/kaggle/input`, and a blocking sleep in my mock server that serialized requests.
- AWQ quantization runs in its own virtualenv (`scripts/quantize_awq.py`) so its pinned dependencies cannot clash with vLLM's.

## Kaggle dry run on the real platform
- The dry run passed on real Kaggle (12/12 smoke, 0 load-test failures). It confirmed that Kaggle extracts an uploaded archive into a READ-ONLY input folder, so the copy-to-writable-folder step is required.
- **Hardening found by reviewing what the real run would measure:** the service's LLM timeout defaults to 30 s, so 50 users on one T4 would silently fall back to extractive answers (HTTP 200, 0 Locust failures) while appearing to measure the LLM. The notebook now sets `LLM_TIMEOUT_S = 120`, reads the API's own `/metrics` after the load test, prints how many answers fell back, and tells you not to report the numbers if any did. Verified with a forced-timeout rehearsal (454/498 fallbacks flagged). It also snapshots GPU memory after the servers start and stores the LLM summary in `run_info.json`.

## Real GPU run on Kaggle
- Executed on 2x Tesla T4 with vLLM 0.31.0: `Qwen/Qwen2.5-1.5B-Instruct` vs the publisher's official AWQ checkpoint (not quantized by us).
- Before/after (16 questions, proxy faithfulness): p50 781 -> 190 ms, tokens/s 48 -> 94, **p95 2372 -> 3292 ms (worse)**, faithfulness proxy 0.52 -> 0.38 (**drop 0.143, over the 0.03 budget**; RAGAS not run).
- Load test, 50 users, 90 s, vLLM in the loop: 1,934 requests, 0 failures, 0 LLM fallbacks (verified from the API's own metrics), p50 880 ms, p95 about 4.1 s; Arabic requests have a long tail.
- Notebook cleaned (debug cells removed, `LD_LIBRARY_PATH` fix kept and documented, `VLLM_VERSION` pinned); results and logs committed under `reports/`.

## CI, alerts and evidence tooling
- CI: explicit "rebuild index" step; new scheduled `monitor.yml` (weekly proxy check; real RAGAS only if `OPENAI_API_KEY` is set, not yet run); CI badge in the README.
- Alerts: `SlowAsk` threshold 1 s -> 5 s (a real LLM measured p95 of about 4 s; 1 s would fire permanently).
- Compose: `api` gets `extra_hosts` for host services; new `docker/docker-compose.demo.yml` overlay (LLM on the host, Langfuse keys, optional assumed price) for producing evidence.
- `docs/evidence-runbook.md`: step-by-step MLflow, Grafana/Prometheus, alert-received and Langfuse screenshots into `reports/screenshots/`. The Docker stacks themselves have not been started by the author.
- Load-test summary rewritten for the LLM run; the extractive-backend test is archived. `reports/final-checklist.md` brought up to date (19 done / 12 partial / 3 not done).
