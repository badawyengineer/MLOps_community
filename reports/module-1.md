# Module 1 — From notebook to production-ready service

**Track:** B — LLM/RAG (Arabic Legal Document Q&A)
**Corpus status:** bootstrap sample corpus (5 hand-verified real articles from Law No. 131 of
1948). The full ~1,150-article corpus ingestion (Step 0, PDF → structured JSON) is the first task
of the next work session — see "Growing the corpus" in the README.

## Baseline

There is no `.pkl`/MAE baseline for this track the way Track A has one — the Track B equivalent
of "the messy notebook" is `sample_corpus()` in `src/prodml/corpus.py` plus the TF-IDF retriever
in `src/prodml/retriever.py`. That pairing *is* the "before" picture: a dependency-light retrieval
baseline with an extractive (non-generative) answer, which Module 2's chunking experiments and
Module 3's real generative model will both be measured against.

**Retrieval smoke test** (5-article sample corpus, char n-gram TF-IDF, cosine similarity):

| Query (Arabic) | Top match | Cosine score |
|---|---|---|
| "العقد شريعة المتعاقدين" | Article 147 | 0.87 (near-verbatim query) |
| "ما هي شروط صحة العقد؟" | Article 147 | 0.31 |
| "zzz gibberish qqq unrelated" | — | 0.0 (correctly returns nothing) |

## Latency (local smoke test, this sandbox — CPU, no containerization)

| Endpoint | Observed latency |
|---|---|
| `/ask` (single question, cold TF-IDF query against 5 articles) | 1.37 ms |
| `/health` | <1 ms |

These numbers are only meaningful as an order-of-magnitude sanity check on a 5-article corpus —
re-measure once the full corpus (~1,150 articles) is loaded, and again once `/ask` calls a real
generative model in Module 3. **Do not compare these against later Locust numbers directly**; the
methodology differs (single hand-timed request vs. a Locust load-test harness).

## Serialization format decision

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load from untrusted source |
|---|---|---|---|---|
| JSON | Yes | Yes | No (unless paired with a schema) | Yes |
| Protobuf | No | Yes | Yes | Yes |
| Pickle | No | No (Python-only) | No | **No — executes arbitrary code on load** |
| ONNX | No | Yes | Yes (via the graph spec) | Mostly (runtime-dependent) |

**Decision:** the corpus and API payloads are JSON (human-readable, cross-language, trivially
debuggable — the right choice for text/metadata that engineers and reviewers need to read
directly). There is no local neural network being pickled in Module 1 for this track (the
retriever is a stateless-at-inference `scikit-learn` TF-IDF matrix, not a model artifact we ship
around); if a reranker or a local embedding model is added in Module 2, it will be evaluated
against this same table before a serialization choice is locked in — pickle would be rejected on
the same "executes arbitrary code" ground the handbook flags. *Never load a `.pkl` you did not
produce yourself.*

## Maturity self-assessment (Google Cloud MLOps maturity model)

**Current level: Level 0 → transitioning to Level 1.**

- Training and "inference" (retrieval) are both scripted and reproducible from the repo, but
  there is no automated pipeline yet — running `prodml-build-corpus` and starting the API are two
  separate manual steps.
- There is exactly one "model" today (the TF-IDF retriever), so there's no experiment tracking or
  model registry pressure yet — that pressure appears the moment Module 2 introduces multiple
  chunking/embedding configurations to compare, which is precisely what MLflow is for.
- CI/CD, data versioning, and continuous training do not exist yet — that is deliberately left to
  Module 2, which is what's missing to reach Level 1.

## Definition of Done

- [x] `pip install -e .` succeeds in a clean virtualenv
- [x] Lint (`ruff check`) and format (`black --check`) both pass
- [x] Tests pass with coverage ≥ 70% (actual: **91.4%**, 34/34 tests passing)
- [x] Zero `print()` statements in `src/` (grep-verified; CLI scripts use `logging` instead)
- [x] `/health`, `/metadata`, `/ask`, `/ask/batch` all respond correctly (curl-verified)
- [ ] Image pushed to Docker Hub and pullable by someone else; container runs as non-root —
      **not yet run**: this sandbox has no Docker daemon. Build and push locally with:
      `docker build -f docker/Dockerfile -t <user>/legal-qa:0.1.0 .`
      `docker run --rm -p 8000:8000 <user>/legal-qa:0.1.0`
      `docker exec <id> whoami` → should print `appuser`
- [ ] Single-stage vs multi-stage image size comparison — capture once the image is built locally
- [x] `README.md` gets a stranger to a prediction in 3 commands
- [x] `reports/module-1.md` has: retrieval smoke test, latency, serialization table, maturity
      self-assessment
- [ ] Pull request opened, reviewed by two peers, merged; `v0.1.0` tagged — pending push to the
      real GitHub remote (this repo was built and tested locally; see README "Pushing this repo")
