# Load test - 50 concurrent users, real LLM in the loop

`locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 90s --csv reports/loadtest/ask --html reports/loadtest/ask.html`

Run on Kaggle (2x Tesla T4). The API used the **vLLM** backend against the baseline model
`Qwen/Qwen2.5-1.5B-Instruct` (served by vLLM 0.31.0, LLM timeout 120 s).

| endpoint | requests | failures | req/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| /ask [abstain] | 179 | 0 | 2.0 | 4 | 11 | 24 |
| /ask [ar] | 893 | 0 | 10.0 | 930 | 4000 | 4200 |
| /ask [en] | 508 | 0 | 5.7 | 930 | 1100 | 1100 |
| /ask/stream | 166 | 0 | 1.9 | 3 | 9 | 17 |
| /health | 188 | 0 | 2.1 | 2 | 4 | 14 |
| Aggregated | 1934 | 0 | 21.7 | 880 | 4000 | 4100 |

**What the LLM actually did** (read from the API's own `/metrics` during the run, see `run_info.json`):
1757 answers, **0 fallbacks to extractive text**, 179 abstentions,
0 unsupported citations, 517682 prompt / 73780 completion tokens,
mean faithfulness proxy 0.548 (lexical proxy, not RAGAS).
Locust exceptions: none (`ask_exceptions.csv` is header-only). Failures: 0.

**Read this before quoting it.**
- Latency here is dominated by the LLM. `/ask [abstain]`, `/ask/stream` and `/health` are fast because they never wait on a full generation.
- `/ask [ar]` has a long tail (p95 about 4.1 s) compared with `/ask [en]` (p95 about 1.1 s). Its median is the same as English, so a
  share of Arabic requests is much slower. The cause was not investigated (longer generations are the likely reason, but this is unverified).
- Locust and the API ran in the same Kaggle session as both vLLM servers; the 90 s run includes a 10 s ramp-up (-r 5).
- One run, one model size (1.5B), a 5-article corpus. Treat these as a mechanism check, not a capacity claim.
- Interactive report: `ask.html`. The earlier extractive-backend test is kept in `SUMMARY_extractive.md`.
