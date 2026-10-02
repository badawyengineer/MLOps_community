"""One-command end-to-end check of the whole service.

    prodml-smoke                          # in-process: no server, no Docker, ~2 seconds
    prodml-smoke --url http://localhost:8000      # against a running server / Docker container

Runs every endpoint, the abstention path, streaming, metrics, guardrails, corpus validation and the
retrieval-quality gate, and prints a PASS/FAIL line for each. Exit code 1 if anything fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

GOOD_Q = "Must contracts be performed in good faith?"
OUT_OF_SCOPE = "كيف أطبخ الكشري؟"


def _checks(client: Any, live: bool) -> list[tuple[str, Callable[[], str]]]:
    def ask(q: str) -> dict:
        r = client.post("/ask", json={"question": q})
        assert r.status_code == 200, f"HTTP {r.status_code}"
        return r.json()

    def health() -> str:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "healthy", r.text
        return f"{r.json()['documents_indexed']} documents indexed"

    def metadata() -> str:
        m = client.get("/metadata").json()
        return f"retriever={m['retriever']} generation={m['generation_backend']}"

    def english() -> str:
        d = ask(GOOD_Q)
        assert not d["abstained"] and d["sources"][0].endswith("Article 148"), d
        return f"cites {d['sources'][0]} (confidence {d['confidence']})"

    def arabic() -> str:
        d = ask("هل يجوز تعديل العقد")
        assert not d["abstained"] and d["sources"], d
        return f"cites {d['sources'][0]}"

    def abstain() -> str:
        d = ask(OUT_OF_SCOPE)
        assert d["abstained"] and d["sources"] == [], d
        return "refused an out-of-scope question instead of guessing"

    def validation() -> str:
        assert client.post("/ask", json={"question": "x"}).status_code == 422
        assert client.post("/ask", json={}).status_code == 422
        return "bad input -> 422"

    def batch() -> str:
        r = client.post("/ask/batch", json={"questions": [GOOD_Q, "من المسؤول عن تعويض الضرر"]})
        assert r.status_code == 200 and len(r.json()["results"]) == 2, r.text
        return "2 results"

    def stream() -> str:
        r = client.post("/ask/stream", json={"question": GOOD_Q})
        events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "meta" and kinds[-1] == "done" and kinds.count("token") > 3, kinds
        return f"{kinds.count('token')} token events then done"

    def metrics() -> str:
        text = client.get("/metrics").text
        needed = [
            "prodml_requests_total",
            "prodml_request_latency_seconds",
            "prodml_abstentions_total",
            "prodml_top_score",
            "prodml_faithfulness_proxy",
            "prodml_query_drift_drop",
        ]
        missing = [n for n in needed if n not in text]
        assert not missing, f"missing series: {missing}"
        return f"{len(needed)} key series exposed"

    checks: list[tuple[str, Callable[[], str]]] = [
        ("GET  /health", health),
        ("GET  /metadata", metadata),
        ("POST /ask (English)", english),
        ("POST /ask (Arabic)", arabic),
        ("POST /ask (out of scope -> abstain)", abstain),
        ("POST /ask (input validation)", validation),
        ("POST /ask/batch", batch),
        ("POST /ask/stream (SSE)", stream),
        ("GET  /metrics (Prometheus)", metrics),
    ]

    if not live:  # these need the code/data on disk, so they only run in-process

        def guardrails() -> str:
            from prodml.guardrails import apply_guardrails, redact_pii

            out, kinds = redact_pii("call 01012345678 or a@b.com")
            assert "01012345678" not in out and {"phone", "email"} <= set(kinds)
            assert apply_guardrails("see [Article 999]", {148}).unsupported_citations == [999]
            return "PII redacted, hallucinated citation caught"

        def corpus() -> str:
            from prodml.config import settings
            from prodml.corpus import load_corpus, sample_corpus
            from prodml.validate_corpus import validate_corpus

            arts = (
                load_corpus(settings.corpus_path)
                if settings.corpus_path.exists()
                else sample_corpus()
            )
            res = validate_corpus(arts)
            assert res.ok, res.errors
            return f"{len(arts)} articles pass validation"

        def quality_gate() -> str:
            from pathlib import Path

            from prodml.config import settings
            from prodml.corpus import load_corpus, sample_corpus
            from prodml.evaluation import check_gates, evaluate, load_eval_set
            from prodml.retriever import build_retriever

            arts = (
                load_corpus(settings.corpus_path)
                if settings.corpus_path.exists()
                else sample_corpus()
            )
            r = evaluate(build_retriever("hybrid").fit(arts), load_eval_set(), k=1, threshold=0.2)
            gates = json.loads(Path("data/eval/thresholds.json").read_text())
            failures = check_gates(r, gates)
            assert not failures, failures
            return f"hit@1={r['hit_at_k']:.2f} MRR={r['mrr']:.2f} correct-abstain={r['correct_abstain_rate']:.2f}"

        checks += [
            ("Guardrails (PII + citations)", guardrails),
            ("Corpus validation", corpus),
            ("Retrieval-quality gate", quality_gate),
        ]
    return checks


def run(client: Any, live: bool) -> int:
    results = []
    for name, fn in _checks(client, live):
        try:
            detail, ok = fn(), True
        except Exception as exc:  # noqa: BLE001 - a smoke test reports every failure, never stops
            detail, ok = f"{type(exc).__name__}: {exc}"[:200], False
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<38} {detail}")
    passed = sum(results)
    print(
        f"\n{passed}/{len(results)} checks passed"
        + ("" if passed == len(results) else "  <-- FAILURES ABOVE")
    )
    return 0 if passed == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", help="test a running server instead of starting one in-process")
    args = ap.parse_args(argv)
    import logging
    import warnings

    logging.disable(
        logging.CRITICAL
    )  # keep the PASS/FAIL table readable; the app's JSON logs are noise here
    warnings.filterwarnings("ignore")
    if args.url:
        import httpx

        print(f"Smoke test against {args.url}\n")
        with httpx.Client(base_url=args.url.rstrip("/"), timeout=30) as client:
            return run(client, live=True)
    from fastapi.testclient import TestClient

    from prodml.api.main import app

    print("Smoke test (in-process - no server needed)\n")
    with TestClient(app) as client:  # `with` runs the app's startup (loads the model)
        return run(client, live=False)


if __name__ == "__main__":
    sys.exit(main())
