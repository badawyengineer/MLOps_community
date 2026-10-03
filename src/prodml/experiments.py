"""Chunking / embedding experiment grid, tracked in MLflow (handbook: >=5 runs varying
chunk_size, overlap, embedding_model, with a quality metric; best run registered).

    python -m prodml.experiments                 # offline embeddings (hashing, lsa)
    python -m prodml.experiments --models intfloat/multilingual-e5-small BAAI/bge-m3

Each run logs params (chunk_size, overlap, embedding_model, k), retrieval metrics at that run's
best abstention threshold, and ``faithfulness_proxy`` (lexical; NOT RAGAS). ``ragas_*`` metrics are
logged only when real RAGAS scores are supplied via ``--ragas-json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from prodml.config import settings
from prodml.corpus import load_corpus, sample_corpus
from prodml.embedders import build_embedder
from prodml.evaluation import best_threshold, load_eval_set, sweep_thresholds
from prodml.faithfulness import faithfulness_proxy
from prodml.retriever import VectorRetriever

CHUNK_GRID = [(0, 0), (200, 0), (200, 50), (400, 50)]  # (chunk_size, overlap); 0 = whole article
REGISTERED_NAME = "legal-qa-retriever"


def run_grid(models: list[str], k: int = 1) -> list[dict[str, Any]]:
    articles = (
        load_corpus(settings.corpus_path) if settings.corpus_path.exists() else sample_corpus()
    )
    items = load_eval_set()
    results = []
    for model, (size, overlap) in itertools.product(models, CHUNK_GRID):
        retriever = VectorRetriever(build_embedder(model), size, overlap).fit(articles)
        sweep = sweep_thresholds(retriever, items, k)
        best = best_threshold(sweep)
        # lexical faithfulness of the extractive answer to its own retrieved context
        proxies = []
        for it in items:
            if it["answerable"]:
                top = retriever.query(it["question"], top_k=1)
                if top:
                    a = top[0].article
                    proxies.append(
                        faithfulness_proxy(a["text_en"], [a["text_en"] + " " + a["text_ar"]])
                    )
        results.append(
            {
                "params": {
                    "embedding_model": model,
                    "chunk_size": size,
                    "overlap": overlap,
                    "k": k,
                    "threshold": best["threshold"],
                    "n_chunks": retriever.store.meta["n_chunks"],
                },
                "metrics": {
                    "hit_at_k": best["hit_at_k"],
                    "mrr": best["mrr"],
                    "false_abstain_rate": best["false_abstain_rate"],
                    "correct_abstain_rate": best["correct_abstain_rate"],
                    "faithfulness_proxy": sum(proxies) / len(proxies) if proxies else 0.0,
                },
            }
        )
    return results


def quality(r: dict[str, Any]) -> float:
    """Single ranking score: half retrieval quality (MRR), half abstention quality (balanced
    accuracy of refusing out-of-scope questions without refusing answerable ones). Ranking on MRR
    alone would crown a model that retrieves well but can never say 'I don't know'."""
    m = r["metrics"]
    return 0.5 * m["mrr"] + 0.5 * (m["correct_abstain_rate"] + 1 - m["false_abstain_rate"]) / 2


def best_of(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest ``quality``; ties go to fewer chunks (a smaller, simpler index)."""
    return max(results, key=lambda r: (round(quality(r), 6), -r["params"]["n_chunks"]))


def log_to_mlflow(results: list[dict[str, Any]], ragas: dict[str, float] | None = None) -> str:
    import mlflow
    from mlflow import MlflowClient

    from prodml.mlflow_model import RetrievalConfigModel

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    name = "legal-qa-chunking-embedding"
    if (
        mlflow.get_experiment_by_name(name) is None
    ):  # artifact root is configurable (tests use a tmp dir)
        mlflow.create_experiment(
            name, artifact_location=os.environ.get("PRODML_MLFLOW_ARTIFACTS") or None
        )
    mlflow.set_experiment(name)
    best = best_of(results)
    best_run_id = ""
    for r in results:
        with mlflow.start_run(
            run_name=f"{r['params']['embedding_model']}-c{r['params']['chunk_size']}-o{r['params']['overlap']}"
        ) as run:
            mlflow.log_params(r["params"])
            mlflow.log_metrics(r["metrics"])
            if ragas and r is best:  # only ever logged when real RAGAS numbers were supplied
                mlflow.log_metrics({f"ragas_{k}": v for k, v in ragas.items()})
            if r is best:
                best_run_id = run.info.run_id
                with tempfile.TemporaryDirectory() as tmp:
                    cfg = Path(tmp) / "config.json"
                    cfg.write_text(json.dumps(r["params"]))
                    corpus = (
                        settings.corpus_path
                        if settings.corpus_path.exists()
                        else _write_sample(Path(tmp))
                    )
                    mlflow.pyfunc.log_model(
                        name="model",
                        python_model=RetrievalConfigModel(),
                        artifacts={"config": str(cfg), "corpus": str(corpus)},
                        registered_model_name=REGISTERED_NAME,
                    )
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(REGISTERED_NAME, "champion", latest.version)
    return best_run_id


def _write_sample(tmp: Path) -> Path:
    p = tmp / "articles.json"
    p.write_text(json.dumps(sample_corpus(), ensure_ascii=False), encoding="utf-8")
    return p


def plot_runs(results: list[dict[str, Any]], out: Path) -> None:
    """Bar chart of all runs (generated from the same numbers logged to MLflow; it is not a
    screenshot of the MLflow UI - take that yourself for the handbook's /reports/ requirement)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{r['params']['embedding_model']}\nc{r['params']['chunk_size']}/o{r['params']['overlap']}"
        for r in results
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.9), 4.5))
    xs = range(len(results))
    ax.bar([x - 0.2 for x in xs], [r["metrics"]["mrr"] for r in results], 0.4, label="MRR")
    ax.bar(
        [x + 0.2 for x in xs],
        [quality(r) for r in results],
        0.4,
        label="quality (MRR + abstention)",
    )
    ax.set_xticks(list(xs), labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Chunking x embedding experiments (offline embeddings, bootstrap eval set)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["hashing-512", "hashing-1024", "lsa-32"])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--plot", type=Path, help="also write a comparison chart PNG here")
    ap.add_argument("--ragas-json", type=Path, help="real RAGAS scores to attach to the best run")
    args = ap.parse_args(argv)

    results = run_grid(args.models, args.k)
    print(
        f"{'embedding':<14}{'chunk':>6}{'ovl':>5}{'chunks':>7}{'hit@k':>7}{'MRR':>6}{'corr-abst':>10}{'thr':>6}{'quality':>8}"
    )
    for r in results:
        p, m = r["params"], r["metrics"]
        print(
            f"{p['embedding_model']:<14}{p['chunk_size']:>6}{p['overlap']:>5}{p['n_chunks']:>7}"
            f"{m['hit_at_k']:>7.2f}{m['mrr']:>6.2f}{m['correct_abstain_rate']:>10.2f}{p['threshold']:>6.2f}{quality(r):>7.2f}"
        )
    if args.plot:
        plot_runs(results, args.plot)
    b = best_of(results)["params"]
    print(f"\nbest: {b['embedding_model']} chunk={b['chunk_size']} overlap={b['overlap']}")
    if not args.no_mlflow:
        ragas = json.loads(args.ragas_json.read_text()) if args.ragas_json else None
        print(
            "MLflow run:",
            log_to_mlflow(results, ragas),
            f"-> registered '{REGISTERED_NAME}' alias @champion",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
