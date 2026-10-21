"""Retrieval + abstention evaluation harness.

Scores a retriever against a hand-labelled question set:

  * answerable questions -> hit@k, recall@k, MRR (is the right article retrieved,
    and how high?), plus how often we wrongly *abstain* on them;
  * unanswerable questions -> how often we correctly abstain instead of
    answering from an irrelevant article.

``abstain`` means: no result, or top score below ``threshold``.

    python -m prodml.evaluation --compare
    python -m prodml.evaluation --retriever hybrid --threshold 0.25 --check data/eval/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prodml._console import ensure_utf8_output
from prodml.config import settings
from prodml.corpus import load_corpus, sample_corpus
from prodml.retriever import RETRIEVERS, Retriever, build_retriever

DEFAULT_EVAL_PATH = Path("data/eval/questions.json")


def load_eval_set(path: Path = DEFAULT_EVAL_PATH) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    for it in items:
        if bool(it["relevant_articles"]) != bool(it["answerable"]):
            raise ValueError(f"{it['id']}: 'answerable' disagrees with 'relevant_articles'")
    return items


def evaluate(
    retriever: Retriever, items: list[dict[str, Any]], k: int = 3, threshold: float = 0.0
) -> dict[str, Any]:
    hits = recall_sum = rr_sum = false_abstain = 0.0
    correct_abstain = 0
    n_ans = n_unans = 0
    misses: list[str] = []
    wrong_answers: list[str] = []

    for it in items:
        results = retriever.query(it["question"], top_k=k)
        abstain = not results or results[0].score < threshold
        if it["answerable"]:
            n_ans += 1
            relevant = set(it["relevant_articles"])
            ranked = [m.article["article_number"] for m in results]
            found = relevant & set(ranked)
            hits += bool(found)
            recall_sum += len(found) / len(relevant)
            rr_sum += next((1 / (r + 1) for r, a in enumerate(ranked) if a in relevant), 0.0)
            false_abstain += abstain
            if not found:
                misses.append(it["id"])
        else:
            n_unans += 1
            correct_abstain += abstain
            if not abstain:
                wrong_answers.append(it["id"])

    return {
        "k": k,
        "threshold": threshold,
        "n_answerable": n_ans,
        "n_unanswerable": n_unans,
        "hit_at_k": hits / n_ans if n_ans else 0.0,
        "recall_at_k": recall_sum / n_ans if n_ans else 0.0,
        "mrr": rr_sum / n_ans if n_ans else 0.0,
        "false_abstain_rate": false_abstain / n_ans if n_ans else 0.0,
        "correct_abstain_rate": correct_abstain / n_unans if n_unans else 0.0,
        "retrieval_misses": misses,
        "answered_out_of_scope": wrong_answers,
    }


def sweep_thresholds(
    retriever: Retriever, items: list[dict[str, Any]], k: int, grid: list[float] | None = None
) -> list[dict[str, Any]]:
    """Abstention trade-off curve: raising the threshold refuses more out-of-scope
    questions but starts refusing answerable ones too."""
    grid = grid if grid is not None else [round(0.05 * i, 2) for i in range(14)]
    return [evaluate(retriever, items, k, t) for t in grid]


def best_threshold(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    """Threshold maximizing balanced accuracy (mean of correct- and non-false-abstain rates);
    ties go to the LOWER threshold, so we never refuse more than we have to."""
    return max(
        sweep,
        key=lambda r: (
            (r["correct_abstain_rate"] + 1 - r["false_abstain_rate"]) / 2,
            -r["threshold"],
        ),
    )


def check_gates(result: dict[str, Any], gates: dict[str, float]) -> list[str]:
    """Return human-readable gate failures (empty list = pass)."""
    failures = []
    for metric, minimum in gates.items():
        if metric.startswith("_"):
            continue
        if metric == "max_false_abstain_rate":
            if result["false_abstain_rate"] > minimum:
                failures.append(
                    f"false_abstain_rate {result['false_abstain_rate']:.3f} > {minimum}"
                )
        elif result[metric] < minimum:
            failures.append(f"{metric} {result[metric]:.3f} < {minimum}")
    return failures


def _load_articles() -> list:
    return load_corpus(settings.corpus_path) if settings.corpus_path.exists() else sample_corpus()


def _log_mlflow(name: str, result: dict[str, Any]) -> None:  # pragma: no cover - optional extra
    import os

    import mlflow

    # MLflow's plain file store is deprecated; default to a local SQLite backend.
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("legal-qa-retrieval")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"retriever": name, "k": result["k"], "threshold": result["threshold"]})
        mlflow.log_metrics(
            {
                key: result[key]
                for key in (
                    "hit_at_k",
                    "recall_at_k",
                    "mrr",
                    "false_abstain_rate",
                    "correct_abstain_rate",
                )
            }
        )


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_output()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_PATH)
    p.add_argument("--retriever", default=settings.retriever, choices=sorted(RETRIEVERS))
    p.add_argument("--compare", action="store_true", help="evaluate every offline retriever")
    p.add_argument("--k", type=int, default=settings.top_k)
    p.add_argument("--threshold", type=float, default=settings.min_confidence)
    p.add_argument("--sweep", action="store_true", help="print the abstention-threshold curve")
    p.add_argument("--check", type=Path, help="JSON of minimum metrics; exit 1 on regression")
    p.add_argument("--out", type=Path, help="write the full JSON report here")
    p.add_argument("--mlflow", action="store_true", help="also log runs to MLflow")
    args = p.parse_args(argv)

    items, articles = load_eval_set(args.eval_set), _load_articles()
    # --compare tables the offline baselines AND the shipped retriever, so the gates (which guard
    # the shipped one only) always have a row to check, even when it is not an offline baseline.
    names = (
        list(dict.fromkeys(["tfidf", "bm25", "hybrid", args.retriever]))
        if args.compare
        else [args.retriever]
    )
    report: dict[str, Any] = {}
    for name in names:
        retriever = build_retriever(name).fit(articles)
        report[name] = evaluate(retriever, items, args.k, args.threshold)
        if args.sweep:
            curve = sweep_thresholds(retriever, items, args.k)
            report[name]["sweep"] = curve
            report[name]["best_threshold"] = best_threshold(curve)["threshold"]
        if args.mlflow:
            _log_mlflow(name, report[name])

    header = f"{'retriever':<8} {'hit@k':>6} {'recall':>7} {'MRR':>6} {'false-abst':>11} {'correct-abst':>13}"
    print(header)
    for name, r in report.items():
        print(
            f"{name:<8} {r['hit_at_k']:>6.2f} {r['recall_at_k']:>7.2f} {r['mrr']:>6.2f} "
            f"{r['false_abstain_rate']:>11.2f} {r['correct_abstain_rate']:>13.2f}"
        )
        if args.sweep:
            print(f"  best threshold (balanced accuracy): {r['best_threshold']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.check:
        # Gates guard the retriever that ships; --compare rows are informational only.
        gates = json.loads(args.check.read_text(encoding="utf-8"))
        shipped = args.retriever
        failed = False
        for msg in check_gates(report[shipped], gates):
            print(f"GATE FAIL [{shipped}]: {msg}", file=sys.stderr)
            failed = True
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
