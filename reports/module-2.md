# Module 2 — Evaluation, retrieval comparison, reproducibility

**Corpus status:** still the 5-article bootstrap sample (4 real articles + 1 repealed placeholder).
Every number below is therefore a **mechanism check, not a quality claim** — see "Limits".

## What was added
- `prodml-eval`: hit@k, recall@k, MRR, correct/false abstention rates, threshold sweep, CI gate.
- Retrievers behind one interface: `tfidf`, `bm25` (own implementation, light Arabic stemming +
  stopwords), `hybrid` (weighted blend, scores kept in [0,1]), `embedding` (implemented, see Limits).
- Abstention: refuse when top score < `PRODML_MIN_CONFIDENCE`; responses carry `confidence`,
  `abstained`, and a not-legal-advice disclaimer.
- Arabic search normalization: tatweel removal, Arabic-Indic and Persian digits, Persian kaf/yeh.
- Prometheus `/metrics`, alert rules (`docker/observability/`), MLflow logging (SQLite backend),
  DVC pipeline (`dvc.yaml`), GitHub Actions CI incl. a retrieval-quality gate.

## Results (23-question bootstrap set: 16 answerable, 7 out-of-scope; k=1, threshold 0.20)

| Retriever | hit@1 | MRR | false-abstain | correct-abstain |
|---|---|---|---|---|
| tfidf  | 0.94 | 0.94 | 0.06 | 0.86 |
| bm25   | 0.88 | 0.88 | 0.00 | 0.86 |
| **hybrid** (default) | **0.94** | **0.94** | **0.00** | **0.86** |

Remaining hybrid errors: one retrieval miss (`a1-2`, an abstract question about the *order* of
legal sources) and one out-of-scope question answered (`x-3`, marriage-contract conditions — it
shares the word "contract" with real civil-code articles; a genuinely hard negative).
hit@3 is 1.00 for all retrievers but means nothing here: k=3 covers most of a 4-article corpus.

## Limits (read before quoting any number)
1. **Tiny corpus.** With 4 real articles, retrieval is nearly trivial. Re-run on the full corpus.
2. **Threshold tuned on the evaluation set** (0.15–0.25 all gave 0 false abstentions for hybrid; 0.20
   picked as the middle). No held-out split exists, so abstention quality is optimistic.
3. **Eval set is small and written by the project author**, so it can share blind spots with the
   system. Expand to 50–100 items, ideally with a second annotator.
4. **`embedding` retriever is unverified against a real model.** The logic is unit-tested with an
   injected encoder; downloading and benchmarking `multilingual-e5-small` (and an Arabic-specific
   model) is the next experiment, logged via `--mlflow`.
5. **Docker image still unbuilt locally**; the CI `docker` job is its first real test.
6. Latency (live service, 5 articles, client-side incl. HTTP): p50 ≈ 3.1 ms, p95 ≈ 3.5 ms. Not
   comparable to later load-test numbers.

## Definition of Done (Module 2, as far as this repo can show)
- [x] Experiments tracked (MLflow, verified writing/reading runs locally)
- [x] Data/pipeline versioned (`dvc repro` verified; second run skips unchanged stages)
- [x] CI written with lint, tests, and a metric-regression gate (gate logic tested; workflow not yet run on GitHub)
- [ ] CI green on GitHub; Docker image built, pushed, non-root verified
- [ ] Full corpus ingested; eval set expanded; embedding models benchmarked
- [ ] PR reviewed by two peers and merged
