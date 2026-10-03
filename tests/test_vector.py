from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from prodml.chunking import chunk_article, chunk_corpus
from prodml.corpus import sample_corpus
from prodml.embedders import HashingEmbedder, LsaEmbedder, build_embedder
from prodml.retriever import DenseHybridRetriever, VectorRetriever, build_retriever
from prodml.vectorstore import VectorStore


@pytest.fixture(scope="module")
def articles():
    return sample_corpus()


def _long_article():
    a = dict(sample_corpus()[1])
    a["text_en"] = " ".join(
        f"Sentence number {i} about the contract and its effects." for i in range(12)
    )
    return a


def test_default_chunking_is_one_chunk_per_article(articles) -> None:
    chunks = chunk_corpus(articles)
    assert len(chunks) == len(articles) and all(c.index == 0 for c in chunks)


def test_long_article_splits_and_every_chunk_keeps_the_citation() -> None:
    a = _long_article()
    chunks = chunk_article(a, max_chars=150, overlap=50)
    assert len(chunks) > 2
    assert {c.article_number for c in chunks} == {147}
    assert all(c.citation == a["citation"] for c in chunks)
    assert [c.chunk_id for c in chunks][:2] == ["147#0", "147#1"]


def test_overlap_repeats_trailing_sentence_into_next_chunk() -> None:
    a = _long_article()
    plain, over = chunk_article(a, 150, 0), chunk_article(a, 150, 60)
    assert len(over) > len(plain)  # overlap means more, partly redundant chunks
    for prev, nxt in pairwise(over[1:]):
        last_sentence = prev.text.split(". ")[-1]
        assert nxt.text.startswith(last_sentence.strip())
    # without overlap, no sentence is repeated across chunks
    sentences = [x for c in plain[1:] for x in c.text.split(". ")]
    assert len(sentences) == len(set(sentences))


@pytest.mark.parametrize(("size", "ov"), [(-1, 0), (100, 100), (100, 150)])
def test_invalid_chunk_params_rejected(size, ov) -> None:
    with pytest.raises(ValueError):
        chunk_article(sample_corpus()[0], size, ov)


@pytest.mark.parametrize("spec", ["hashing", "hashing-128", "lsa", "lsa-8"])
def test_embedders_return_unit_vectors(spec, articles) -> None:
    emb = build_embedder(spec)
    texts = [a["text_en"] for a in articles]
    emb.fit(texts)
    vecs = emb.encode(texts)
    assert vecs.shape[0] == len(texts)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_unknown_embedder_rejected() -> None:
    with pytest.raises(ValueError):
        build_embedder("nonsense")


def test_lsa_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        LsaEmbedder().encode(["x"])


def test_vector_store_roundtrip_has_no_pickle(tmp_path) -> None:
    store = VectorStore(np.eye(3), [{"a": i} for i in range(3)], {"k": "v"})
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.meta == {"k": "v"} and loaded.search(np.array([0, 1.0, 0]), 1)[0][0] == {"a": 1}
    with pytest.raises(ValueError):
        VectorStore(np.eye(2), [{}], {})


def test_vector_retriever_finds_article_and_scores_bounded(articles) -> None:
    r = VectorRetriever(HashingEmbedder(1024)).fit(articles)
    top = r.query("Must contracts be performed in good faith?", 3)
    assert top[0].article["article_number"] == 148
    assert all(0 <= m.score <= 1 for m in top)


def test_persisted_index_loads_and_matches_fresh_fit(tmp_path, articles) -> None:
    fresh = VectorRetriever(HashingEmbedder(256)).fit(articles)
    fresh.save(tmp_path)
    loaded = VectorRetriever(HashingEmbedder(256)).load(tmp_path, articles)
    q = "هل يجوز تعديل العقد"
    assert [m.article["article_number"] for m in loaded.query(q)] == [
        m.article["article_number"] for m in fresh.query(q)
    ]


def test_stale_index_is_rejected(tmp_path, articles) -> None:
    VectorRetriever(HashingEmbedder(256)).fit(articles).save(tmp_path)
    with pytest.raises(ValueError, match="stale"):
        VectorRetriever(HashingEmbedder(256)).load(tmp_path, articles[:3])  # different corpus
    with pytest.raises(ValueError, match="stale"):
        VectorRetriever(HashingEmbedder(512)).load(tmp_path, articles)  # different config


def test_fitted_embedder_cannot_be_persisted(tmp_path, articles) -> None:
    with pytest.raises(ValueError, match="cannot be persisted"):
        VectorRetriever(LsaEmbedder(8)).fit(articles).save(tmp_path)


def test_dense_hybrid_and_factory(articles) -> None:
    r = DenseHybridRetriever(HashingEmbedder(512)).fit(articles)
    assert r.query("good faith", 1)[0].article["article_number"] == 148
    assert isinstance(build_retriever("vector"), VectorRetriever)
    assert isinstance(build_retriever("hybrid-dense"), DenseHybridRetriever)


def test_index_cli_builds_and_persists(tmp_path, monkeypatch) -> None:
    from prodml import indexing
    from prodml.config import settings
    from prodml.corpus import write_corpus

    monkeypatch.setattr(settings, "index_dir", tmp_path / "idx")
    monkeypatch.setattr(settings, "corpus_path", tmp_path / "articles.json")
    write_corpus(sample_corpus(), settings.corpus_path)
    assert indexing.main([]) == 0
    assert (tmp_path / "idx" / "vectors.npz").exists() and (tmp_path / "idx" / "meta.json").exists()


def test_indexing_refuses_to_silently_use_the_sample_when_corpus_is_missing(
    tmp_path, monkeypatch
) -> None:
    from prodml import indexing
    from prodml.config import settings

    monkeypatch.setattr(settings, "index_dir", tmp_path / "idx")
    monkeypatch.setattr(settings, "corpus_path", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="No corpus"):
        indexing.build_index()
    assert indexing.build_index(allow_sample=True).store.meta["n_articles"] == len(sample_corpus())


def test_corpus_json_is_not_gitignored_so_the_docker_build_and_reviewers_get_it() -> None:
    """Regression: Module 1's .gitignore silently excluded the corpus, breaking `COPY data/processed/`."""
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("git") is None or not (Path(__file__).parents[1] / ".git").exists():
        pytest.skip("not a git checkout")
    r = subprocess.run(
        ["git", "check-ignore", "-q", "data/processed/articles.json"],
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert (
        r.returncode == 1
    ), "data/processed/articles.json is ignored by git - it must be committed"
