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
median 880 ms, p95 about 4.1 s. English `/ask` p95 is about 1.1 s; Arabic `/ask` has a long tail (p95 about 4.1 s). Details: `reports/loadtest/SUMMARY.md`.
