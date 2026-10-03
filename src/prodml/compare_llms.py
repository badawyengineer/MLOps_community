"""Before/after comparison of two OpenAI-compatible LLM endpoints - the table behind the
handbook's quantization deliverable (AWQ 4-bit vs the unquantized baseline, both served by vLLM).

    python -m prodml.compare_llms \
        --a-url http://localhost:8001/v1 --a-model Qwen/Qwen2.5-1.5B-Instruct \
        --b-url http://localhost:8002/v1 --b-model ./qwen-awq --out reports/optimization.md

Retrieval is shared, so any difference comes from the generator alone. Faithfulness here is the
lexical PROXY (see faithfulness.py); for the handbook's "RAGAS faithfulness drop <= 0.03" claim, run
RAGAS yourself on the saved per-question answers (--answers-json). Needs GPU endpoints for a
meaningful result; the harness itself is tested with mocked endpoints.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from prodml._console import ensure_utf8_output
from prodml.config import settings
from prodml.evaluation import load_eval_set
from prodml.faithfulness import faithfulness_proxy
from prodml.llm import LLMError, OpenAICompatClient, build_messages
from prodml.rag import LegalQAPredictor, _build_context


def run_endpoint(
    client: OpenAICompatClient, predictor: LegalQAPredictor, items: list[dict[str, Any]]
) -> dict[str, Any]:
    latencies, tps, proxies, answers = [], [], [], []
    for it in (i for i in items if i["answerable"]):
        r = predictor._retrieve(it["question"])
        if r.abstained:
            continue
        ctx = _build_context(r.matches, settings.max_context_chars)
        t0 = time.perf_counter()
        try:
            res = client.complete(build_messages(it["question"], ctx))
        except LLMError:
            continue
        dt = time.perf_counter() - t0
        latencies.append(dt * 1000)
        if res.completion_tokens and dt > 0:
            tps.append(res.completion_tokens / dt)
        proxies.append(
            faithfulness_proxy(
                res.text, predictor.contexts_for([m.article["citation"] for m in r.matches])
            )
        )
        answers.append({"question": it["question"], "answer": res.text})
    lat = sorted(latencies)
    return {
        "n": len(lat),
        "latency_p50_ms": statistics.median(lat) if lat else 0.0,
        "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0,
        "tokens_per_s": statistics.mean(tps) if tps else 0.0,
        "faithfulness_proxy": statistics.mean(proxies) if proxies else 0.0,
        "answers": answers,
    }


def render_table(a: dict[str, Any], b: dict[str, Any], a_name: str, b_name: str) -> str:
    drop = a["faithfulness_proxy"] - b["faithfulness_proxy"]
    verdict = "within" if drop <= 0.03 else "EXCEEDS"
    rows = [
        ("latency p50 (ms)", a["latency_p50_ms"], b["latency_p50_ms"]),
        ("latency p95 (ms)", a["latency_p95_ms"], b["latency_p95_ms"]),
        ("tokens / s", a["tokens_per_s"], b["tokens_per_s"]),
        ("faithfulness (proxy)", a["faithfulness_proxy"], b["faithfulness_proxy"]),
    ]
    lines = [f"| metric | A: {a_name} | B: {b_name} |", "|---|---|---|"]
    lines += [f"| {m} | {x:.2f} | {y:.2f} |" for m, x, y in rows]
    lines.append(
        f"\nFaithfulness-proxy drop A->B: {drop:+.3f} ({verdict} the 0.03 budget). n={a['n']}/{b['n']} questions."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for side in "ab":
        ap.add_argument(f"--{side}-url", required=True)
        ap.add_argument(f"--{side}-model", required=True)
        ap.add_argument(f"--{side}-key", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--answers-json", type=Path, help="save per-question answers for a real RAGAS pass"
    )
    args = ap.parse_args(argv)

    predictor, items = LegalQAPredictor().load(), load_eval_set()
    a = run_endpoint(OpenAICompatClient(args.a_url, args.a_model, args.a_key), predictor, items)
    b = run_endpoint(OpenAICompatClient(args.b_url, args.b_model, args.b_key), predictor, items)
    table = render_table(a, b, args.a_model, args.b_model)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
    if args.answers_json:
        args.answers_json.write_text(
            json.dumps({"A": a["answers"], "B": b["answers"]}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
