# Evidence runbook: MLflow, Grafana, Langfuse screenshots and the alert test

Everything here is code/config that was validated offline (YAML/JSON parse, unit tests, mock servers). The Docker
stacks themselves were **not** started by the author, so the screenshots below must be taken on your machine.
Save them in `reports/screenshots/` with the file names given, commit, then flip the matching checklist items.

Windows: use Git Bash for the commands, and Docker Desktop (WSL2 backend) for Parts B and C.

---

## Part A - MLflow (no Docker needed, ~5 min)

```bash
pip install -e ".[dev,tracking]"
prodml-experiments --plot reports/mlflow_runs_comparison.png     # >= 5 runs + registers legal-qa-retriever@champion
mlflow ui --backend-store-uri sqlite:///mlflow.db                 # open http://127.0.0.1:5000
```

Screenshots:
1. **`reports/screenshots/mlflow-runs.png`** - the experiment's runs table with 5+ runs, showing the `chunk_size`, `overlap`,
   `embedding_model` params and the `faithfulness_proxy` / `hit_at_k` metric columns (use *Columns* to show them; or select
   the runs and click *Compare*).
2. **`reports/screenshots/mlflow-registry.png`** - *Models* -> `legal-qa-retriever`, showing the **champion** alias.

`mlflow.db` and `mlruns/` are git-ignored on purpose; commit only the screenshots.

## Part B - Grafana + Prometheus + the alert test (Docker, ~15 min)

```bash
# terminal 1: a stand-in LLM on the host (no GPU needed; the numbers are a stub's, not a model's)
prodml-mock-llm --port 8001 --model Qwen/Qwen2.5-1.5B-Instruct --latency-ms 60

# terminal 2: the stack (api + prometheus + pushgateway + alertmanager + grafana)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.yml --profile observability up --build

# terminal 3: traffic for ~3 minutes, then push the eval-set faithfulness
pip install -e ".[loadtest]"
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 10 -r 5 -t 3m
prodml-monitor --push http://localhost:9091
```

Open Grafana at http://localhost:3001 (admin / admin) -> dashboard **Legal QA - RAG health**, time range *Last 15 minutes*.
Wait ~30 s so rate() panels fill in.

3. **`reports/screenshots/grafana-dashboard.png`** - the whole dashboard. Every panel should show data. The *LLM cost / hour*
   panel is 0 for a self-hosted model unless you export an assumed price (see the comment in `docker-compose.demo.yml`);
   if you do, say in the report that the price is assumed.
4. **`reports/screenshots/prometheus-alerts.png`** - http://localhost:9090/alerts showing the rules loaded.

Alert -> notification test (closes "Alert triggered"):
```bash
# terminal 4
prodml-alert-sink                                              # listens on :5001, appends to reports/alerts.jsonl
# terminal 5
prodml-alert-sink --fire-test-alert http://localhost:9093      # sends a FaithfulnessLow alert to Alertmanager
```
5. **`reports/screenshots/alert-received.png`** - terminal 4 printing `[FIRING] FaithfulnessLow (critical)...` and/or the
   Alertmanager UI at http://localhost:9093. Also commit `reports/alerts.jsonl`.
   For a *natural* firing: the proxy faithfulness is ~1.0, so the rule won't fire by itself; the test alert is the evidence.

Note on `SlowAsk`: its threshold is now 5 s (was 1 s). With a real LLM the Kaggle load test measured p95 of about 4 s at
50 users, so 1 s would fire permanently. The dashboard here uses the mock LLM (60 ms), so it will not trigger.

## Part C - Langfuse traces (Docker, ~20 min, needs several GB of RAM)

Langfuse runs in its own compose project (Postgres, ClickHouse, Redis, MinIO):
```bash
git clone https://github.com/langfuse/langfuse ../langfuse
cd ../langfuse && docker compose up -d          # wait until http://localhost:3000 loads (can take a few minutes)
```
1. Open http://localhost:3000, sign up (local only), create an organisation and a project, then *Settings -> API keys ->
   Create* and copy the public (`pk-lf-...`) and secret (`sk-lf-...`) keys. **Never commit these keys.**
2. Restart the API with the keys (same terminals as Part B; stop the old compose first with Ctrl+C):
```bash
export PRODML_LANGFUSE_HOST=http://host.docker.internal:3000
export PRODML_LANGFUSE_PUBLIC_KEY=pk-lf-...
export PRODML_LANGFUSE_SECRET_KEY=sk-lf-...
docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.yml --profile observability up --build
prodml-ask "Must contracts be performed in good faith?"      # or curl -X POST localhost:8000/ask ...
```
3. In Langfuse -> *Tracing* -> open one trace. **`reports/screenshots/langfuse-trace.png`** must show the `retrieve` span,
   the `generation` (with token usage) and the `faithfulness_proxy` score.

If no traces appear: check the API logs for Langfuse errors (tracing never fails a request, so errors are only logged),
and confirm the host URL is reachable from inside the container (`docker compose exec api python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:3000').status)"`).

## Part D - After the screenshots

```bash
git add reports/screenshots reports/alerts.jsonl
git commit -m "Add MLflow, Grafana, Prometheus alert and Langfuse evidence"
git push
```
Then tell the assistant which parts worked, and what each one printed if something failed. Checklist items to flip:
MLflow screenshot (Part A), Grafana cost/hour (Part B), Alert triggered (Part B), Langfuse traces (Part C).
