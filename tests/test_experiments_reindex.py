from __future__ import annotations

import json

import pytest

from prodml import experiments as ex
from prodml import reindex as ri
from prodml.config import settings
from prodml.corpus import sample_corpus, write_corpus
from prodml.retriever import VectorRetriever


def test_grid_has_at_least_five_runs_varying_all_three_knobs() -> None:
    results = ex.run_grid(["hashing-256", "lsa-8"])
    assert len(results) >= 5
    params = [r["params"] for r in results]
    assert len({p["embedding_model"] for p in params}) >= 2
    assert len({p["chunk_size"] for p in params}) >= 2 and len({p["overlap"] for p in params}) >= 2
    assert all({"hit_at_k", "mrr", "faithfulness_proxy"} <= set(r["metrics"]) for r in results)


def test_ranking_penalizes_models_that_cannot_abstain() -> None:
    retrieves_but_never_refuses = {
        "params": {"n_chunks": 5},
        "metrics": {"mrr": 0.95, "correct_abstain_rate": 0.0, "false_abstain_rate": 0.0},
    }
    balanced = {
        "params": {"n_chunks": 5},
        "metrics": {"mrr": 0.85, "correct_abstain_rate": 0.86, "false_abstain_rate": 0.0},
    }
    assert ex.best_of([retrieves_but_never_refuses, balanced]) is balanced


def test_ties_prefer_fewer_chunks() -> None:
    m = {"mrr": 0.9, "correct_abstain_rate": 0.9, "false_abstain_rate": 0.0}
    small, big = {"params": {"n_chunks": 5}, "metrics": m}, {
        "params": {"n_chunks": 50},
        "metrics": m,
    }
    assert ex.best_of([big, small]) is small


def test_mlflow_logging_and_registry_roundtrip(tmp_path, monkeypatch) -> None:
    pytest.importorskip("mlflow")
    import mlflow
    import pandas as pd

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    monkeypatch.setenv("PRODML_MLFLOW_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        settings, "corpus_path", tmp_path / "missing.json"
    )  # forces sample-corpus artifact
    results = ex.run_grid(["hashing-256"])
    ex.log_to_mlflow(results)
    runs = mlflow.search_runs(experiment_names=["legal-qa-chunking-embedding"])
    assert len(runs) == len(results) >= 4
    model = mlflow.pyfunc.load_model(f"models:/{ex.REGISTERED_NAME}@champion")
    out = model.predict(pd.DataFrame({"question": ["Must contracts be performed in good faith?"]}))
    assert out["citation"][0].endswith("Article 148")


@pytest.fixture
def corpus_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "corpus_path", tmp_path / "articles.json")
    monkeypatch.setattr(settings, "index_dir", tmp_path / "index")
    write_corpus(sample_corpus(), settings.corpus_path)
    return tmp_path


def _new_article(n: int, text: str) -> dict:
    return {
        "article_number": n,
        "book": "",
        "chapter": "",
        "section": "",
        "topic": "Sale",
        "text_ar": "البيع عقد يلتزم به البائع",
        "text_en": text,
        "is_repealed": False,
        "source_page": None,
        "citation": f"Egyptian Civil Code, Article {n}",
    }


def test_reindex_adds_one_new_document_and_it_becomes_retrievable(corpus_env) -> None:
    new = corpus_env / "new.json"
    new.write_text(
        json.dumps(
            [
                _new_article(
                    418, "Sale is a contract whereby the seller transfers ownership of a thing."
                )
            ]
        ),
        encoding="utf-8",
    )
    assert ri.main([str(new)]) == 0
    articles = json.loads(settings.corpus_path.read_text(encoding="utf-8"))
    assert (
        418 in {a["article_number"] for a in articles} and len(articles) == len(sample_corpus()) + 1
    )
    from prodml.embedders import build_embedder

    r = VectorRetriever(
        build_embedder(settings.embedding_model), settings.chunk_chars, settings.chunk_overlap
    )
    r.load(
        settings.index_dir, articles
    )  # fingerprint matches -> index was rebuilt for the new corpus
    assert (
        r.query("sale transfers ownership of a thing seller", 1)[0].article["article_number"] == 418
    )


def test_reindex_replaces_an_amended_article_in_place(corpus_env) -> None:
    new = corpus_env / "amend.json"
    new.write_text(
        json.dumps([_new_article(148, "AMENDED text of article 148.")]), encoding="utf-8"
    )
    changes = ri.reindex(new, dry_run=True)
    assert changes == {"added": [], "replaced": [148]}
    assert "AMENDED" not in settings.corpus_path.read_text(
        encoding="utf-8"
    )  # dry run wrote nothing


def test_reindex_refuses_invalid_input_and_writes_nothing(corpus_env) -> None:
    before = settings.corpus_path.read_text(encoding="utf-8")
    bad = corpus_env / "bad.json"
    bad.write_text(
        json.dumps([{**_new_article(500, "x"), "text_ar": ""}]), encoding="utf-8"
    )  # empty Arabic text
    assert ri.main([str(bad)]) == 1
    assert ri.main([str(corpus_env / "nope.json")]) == 1
    assert settings.corpus_path.read_text(encoding="utf-8") == before
