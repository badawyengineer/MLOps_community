# Load test - 50 concurrent users

`locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 25 -t 25s --csv reports/loadtest/ask --html reports/loadtest/ask.html`

| endpoint | requests | failures | req/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| /ask [abstain] | 97 | 0 | 3.9 | 4 | 590 | 710 |
| /ask [ar] | 543 | 0 | 21.9 | 5 | 110 | 710 |
| /ask [en] | 311 | 0 | 12.6 | 5 | 93 | 650 |
| /ask/stream | 84 | 0 | 3.4 | 3 | 83 | 870 |
| /health | 94 | 0 | 3.8 | 2 | 480 | 510 |
| Aggregated | 1129 | 0 | 45.6 | 5 | 110 | 680 |

**Read this before quoting it.**
- Backend was the **extractive** generator (no LLM call), single uvicorn worker, 5-article corpus.
  A real LLM will dominate latency; re-run this test against vLLM and keep both reports.
- Locust and the server shared one small CPU, and the 25 s run includes the ramp-up (all 50 users
  spawn in 2 s), which is why p95/p99 are far above the p50.
- An earlier run reported a clean pass while 119 stream tasks had crashed inside the locustfile
  (a missing `catch_response=True`). It was discarded and re-run after the fix; the exceptions CSV
  for this run is header-only and the stream task validates that a `done` event arrives.
- Full interactive report: `ask.html`.
