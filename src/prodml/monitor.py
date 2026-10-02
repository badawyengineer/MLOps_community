"""Scheduled quality monitor: run the evaluation questions through the live pipeline, score
faithfulness, and push the result to a Prometheus Pushgateway so Grafana can chart it and the
``FaithfulnessLow`` alert (< 0.80) can fire.

    python -m prodml.monitor                              # offline lexical proxy, prints + writes report
    python -m prodml.monitor --push http://localhost:9091 # push gauges to Pushgateway
    python -m prodml.monitor --ragas --push ...           # real RAGAS faithfulness (judge LLM, costs money)

The ``source`` label says which scorer produced the number; the proxy is never reported as RAGAS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from prodml.evaluation import load_eval_set
from prodml.rag import LegalQAPredictor


def collect(
    predictor: LegalQAPredictor, items: list[dict[str, Any]], use_ragas: bool = False
) -> dict[str, Any]:
    rows, proxies, abstained = [], [], 0
    answerable = [i for i in items if i["answerable"]]
    for it in answerable:
        res = predictor.query_one(it["question"])
        if res["abstained"]:
            abstained += 1
            continue
        contexts = predictor.contexts_for(res["sources"])
        proxies.append(res["faithfulness_proxy"] or 0.0)
        rows.append(
            {
                "question": it["question"],
                "answer": res["answer"],
                "contexts": contexts,
                "ground_truth": contexts[0] if contexts else "",
            }
        )  # reference = the cited article itself
    report: dict[str, Any] = {
        "n_questions": len(answerable),
        "n_answered": len(rows),
        "false_abstain_rate": abstained / len(answerable) if answerable else 0.0,
        "faithfulness": sum(proxies) / len(proxies) if proxies else 0.0,
        "source": "proxy",
    }
    if use_ragas:  # pragma: no cover - needs ragas + judge LLM
        from prodml.faithfulness import ragas_scores

        scores = ragas_scores(rows)
        report.update({"source": "ragas", "faithfulness": scores["faithfulness"], "ragas": scores})
    return report


def push(report: dict[str, Any], gateway: str, job: str = "prodml-eval") -> CollectorRegistry:
    reg = CollectorRegistry()
    Gauge(
        "prodml_eval_faithfulness", "Faithfulness on the eval set", ["source"], registry=reg
    ).labels(report["source"]).set(report["faithfulness"])
    Gauge(
        "prodml_eval_false_abstain_rate",
        "Answerable eval questions the system refused",
        registry=reg,
    ).set(report["false_abstain_rate"])
    for name, value in (report.get("ragas") or {}).items():
        Gauge(f"prodml_eval_ragas_{name}", f"RAGAS {name} on the eval set", registry=reg).set(value)
    push_to_gateway(gateway, job=job, registry=reg)
    return reg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval-set", type=Path, default=Path("data/eval/questions.json"))
    ap.add_argument("--push", metavar="GATEWAY_URL")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("reports/monitoring/latest.json"))
    ap.add_argument("--min-faithfulness", type=float, help="exit 1 below this (CI/cron gate)")
    args = ap.parse_args(argv)

    report = collect(LegalQAPredictor().load(), load_eval_set(args.eval_set), args.ragas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.push:
        push(report, args.push)
        print(f"pushed to {args.push}")
    if args.min_faithfulness is not None and report["faithfulness"] < args.min_faithfulness:
        print(
            f"GATE FAIL: faithfulness {report['faithfulness']:.3f} < {args.min_faithfulness}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
