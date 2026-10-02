# Arabic Legal Document Q&A

A retrieval-augmented Q&A service over the Egyptian Civil Code (Law No. 131 of 1948) — the
Track B final project for **The MLOps Practitioner** (ITI x MLOps MENA Community). See
[`reports/track-decision.md`](reports/track-decision.md) for why this track, and the handbook's
"Final Project — Track B" section for the full spec this repo builds toward.

**Status: Module 1 complete; Module 2 work (evaluation, retrieval comparison, CI, versioning,
tracking, first observability pieces) is on branch `module-2-eval-and-retrieval`.** Modules 3–5
(real generation/serving, optimization, full observability) are not started.

## Quickstart (3 commands)

```bash
pip install -e .
python -m prodml.corpus                          # writes the bootstrap sample corpus
uvicorn prodml.api.main:app --reload --port 8000  # serve at http://localhost:8000
```

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "العقد شريعة المتعاقدين"}'
```

```json
{
  "answer": "[Egyptian Civil Code, Article 147] The contract makes the law of the parties. It can be revoked or altered only by mutual consent of both parties, or for reasons provided for by law.",
  "sources": ["Egyptian Civil Code, Article 147", "Egyptian Civil Code, Article 1", "Egyptian Civil Code, Article 148"],
  "scores": [0.3174, 0.1131, 0.0829],
  "model_version": "0.1.0",
  "correlation_id": "9b7438d302ff4f5a",
  "latency_ms": 1.37
}
```

## What's actually here right now vs. what's a documented placeholder

Being direct about this matters more than it looks like it does — the handbook's honesty clause
applies to the README too, not just the final report.

| Piece | Status |
|---|---|
| Corpus | **Sample only** — 5 hand-verified real articles (1, 60 [repealed], 147, 148, 163). The full ~1,150-article corpus needs the actual Civil Code PDF ingested (see below). |
| Retrieval | **Real, working, benchmarked** — `tfidf`, `bm25`, `hybrid` (default) all run offline; `embedding` (multilingual sentence-transformers) is implemented and unit-tested with an injected encoder but **has not been run against a real model** (no model download in the build environment). |
| Abstention | **Real, working, tuned on a tiny set** — answers are refused below a confidence threshold (`PRODML_MIN_CONFIDENCE`, default 0.2). Calibrated on 23 bootstrap questions, not validated on held-out data. |
| Evaluation | **Real, working** — `prodml-eval` (hit@k, recall@k, MRR, abstention rates, threshold sweep, CI gate, optional MLflow). Eval set is a 23-question *bootstrap*; expand it before trusting any number. |
| Generation | **Extractive placeholder** — `/ask` returns the top-matched article's own text, not a generated answer. `LegalQAPredictor._generate()` in `src/prodml/rag.py` is the one method Module 3 replaces. |
| API, logging, metrics, tests | **Real, working, tested** — 66 tests, 94% coverage; `/metrics` exposes Prometheus counters/histograms. |
| CI / data pipeline | `dvc repro` verified locally. GitHub Actions workflow (lint, tests, retrieval-quality gate, Docker build+run+non-root check) is written and YAML-validated but **has not yet run on GitHub**. |
| Docker | **Not built in the environment this was developed in** (no Docker daemon). The CI `docker` job is the first real verification. |

## Growing the corpus (the real next task)

1. Obtain the bilingual Arabic/English Egyptian Civil Code PDF and place it at
   `data/raw/egyptian_civil_code.pdf`.
2. Install the optional PDF extra: `pip install -e ".[pdf]"`.
3. Implement the extraction loop in `build_from_pdf()` (`src/prodml/corpus.py`) against that
   PDF's actual layout — the function's docstring lists the four failure modes to expect
   (column bleed, RTL glyph reversal, Arabic-Indic numerals, repealed-article gaps) and is
   deliberately left as `NotImplementedError` rather than a guess at a layout we haven't seen.
4. Run `prodml-build-corpus`, then `python -m prodml.validate_corpus` — fix everything it flags
   before indexing anything.
5. Spot-check ~20 random articles by eye. Do not trust automated extraction blind on RTL text.

## What's verified in this repo right now

```bash
pip install -e ".[dev]"                                   # install
ruff check src tests && black --check src tests           # lint       (passing)
pytest -v                                                  # test       (66 passed, 94% coverage)
python -m prodml.corpus                                    # build corpus
python -m prodml.validate_corpus                           # validate corpus
uvicorn prodml.api.main:app --reload --port 8000           # serve
```

Docker build/run was **not** exercised in the environment this repo was scaffolded in (no Docker
daemon available there) — build and verify it on your own machine before opening the PR:

```bash
docker build -f docker/Dockerfile -t <your-dockerhub-user>/legal-qa:0.1.0 .
docker run --rm -p 8000:8000 <your-dockerhub-user>/legal-qa:0.1.0
docker exec <container-id> whoami   # should print: appuser
docker push <your-dockerhub-user>/legal-qa:0.1.0
```

## Evaluation and quality gates

```bash
python -m prodml.evaluation --compare --k 1 --threshold 0.2     # tfidf vs bm25 vs hybrid
python -m prodml.evaluation --retriever hybrid --sweep          # abstention-threshold curve
python -m prodml.evaluation --retriever hybrid --k 1 --threshold 0.2 \
    --check data/eval/thresholds.json                           # exits 1 on regression (CI)
pip install -e ".[tracking]" && python -m prodml.evaluation --compare --mlflow   # log to MLflow
dvc repro                                                       # corpus -> validate -> evaluate
```

Questions live in `data/eval/questions.json` (Arabic + English, paraphrases, and out-of-scope
questions the service should *refuse*). Results and caveats: `reports/module-2.md`.

## API

| Endpoint | Method | Behaviour |
|---|---|---|
| `/health` | GET | 200 only if the retriever is actually loaded in memory |
| `/metadata` | GET | model version, articles indexed, generation backend, load timestamp |
| `/ask` | POST | `{"question": str}` → answer, cited sources, `confidence`, `abstained`, disclaimer |
| `/metrics` | GET | Prometheus metrics (requests, latency, abstentions, top-score histogram) |
| `/ask/batch` | POST | `{"questions": [str, ...]}` → list of the same |

Every response carries `correlation_id` (also echoed as the `X-Request-ID` header), and every log
line in the service is structured JSON tagged with that same ID — see `src/prodml/logging_conf.py`.

## Repository layout

```
src/prodml/
├── config.py            # env-driven settings, no hardcoded paths
├── logging_conf.py       # structured JSON logs + correlation-id contextvar
├── corpus.py             # Step 0: PDF -> JSON ingestion, Arabic normalization, sample corpus
├── validate_corpus.py    # corpus validation, run before anything gets embedded
├── retriever.py          # tfidf / bm25 / hybrid / embedding retrievers behind one interface
├── evaluation.py         # hit@k, MRR, abstention metrics, threshold sweep, CI gate, MLflow
├── metrics.py            # Prometheus metrics (incl. retrieval-confidence drift signals)
├── rag.py                # LegalQAPredictor — the one seam Modules 3-5 build on top of
└── api/
    ├── main.py            # FastAPI app, lifespan model loading, correlation-id middleware
    └── schemas.py          # Pydantic request/response models
data/eval/                # labelled eval questions + CI regression thresholds
dvc.yaml                  # reproducible pipeline: build -> validate -> evaluate
.github/workflows/ci.yml  # lint, tests, retrieval gate, Docker verification
tests/                    # pytest suite, 94% coverage, fixtures + mocks + parametrization
docker/                   # multi-stage Dockerfile + docker-compose.yml
reports/
├── track-decision.md      # why Track B
└── module-1.md            # this module's numbers + Definition of Done
```

## Pushing this repo to GitHub

This repo was built and fully tested locally. To get it onto GitHub and open the Module 1 PR:

```bash
cd mlops-legal-qa
git remote add origin git@github.com:<your-username>/mlops-legal-qa.git
git push -u origin main
git push -u origin module-1-packaging
# open a PR: module-1-packaging -> main on GitHub, paste in the Definition of Done from
# reports/module-1.md, request two reviews per the collaboration protocol
# after merge:
git checkout main && git pull
git tag -a v0.1.0 -m "Module 1: packaged RAG service, tested, containerized"
git push --tags
```
