from __future__ import annotations

import numpy as np
import pytest

from prodml.corpus import normalize_for_search, sample_corpus
from prodml.retriever import (
    Bm25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
    build_retriever,
    tokenize,
)


@pytest.fixture(scope="module")
def articles():
    return sample_corpus()


@pytest.mark.parametrize("name", ["tfidf", "bm25", "hybrid"])
def test_every_retriever_finds_good_faith_article(name, articles) -> None:
    top = build_retriever(name).fit(articles).query("must contracts be performed in good faith?", 1)
    assert top[0].article["article_number"] == 148


@pytest.mark.parametrize("name", ["tfidf", "bm25", "hybrid"])
def test_scores_are_bounded_zero_to_one(name, articles) -> None:
    for m in build_retriever(name).fit(articles).query("هل يجوز تعديل العقد", 5):
        assert 0.0 <= m.score <= 1.0


def test_unknown_retriever_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever("nope")


def test_bm25_returns_nothing_for_unseen_terms(articles) -> None:
    assert Bm25Retriever().fit(articles).query("zzz qqq", 3) == []


def test_bm25_and_hybrid_require_fit() -> None:
    with pytest.raises(RuntimeError):
        Bm25Retriever().query("x")
    with pytest.raises(RuntimeError):
        EmbeddingRetriever(encoder=lambda t: np.ones((len(t), 2))).query("x")


def test_hybrid_alpha_extremes_match_components(articles) -> None:
    q = "good faith"
    only_bm25 = HybridRetriever(alpha=0.0).fit(articles).query(q, 1)[0]
    bm25 = Bm25Retriever().fit(articles).query(q, 1)[0]
    assert only_bm25.article["article_number"] == bm25.article["article_number"]
    assert only_bm25.score == pytest.approx(bm25.score)


def test_tokenizer_strips_stopwords_and_definite_article() -> None:
    assert tokenize("ما هي شروط العقد") == ["شروط", "عقد"]
    assert "the" not in tokenize("the contract")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("المـــادة", "الماده"),  # tatweel + ta-marbuta
        ("المادة ١٤٧", "الماده 147"),  # Arabic-Indic digits
        ("المادة ۱۴۷", "الماده 147"),  # Persian digits
        ("أحكام إلغاء", "احكام الغاء"),  # alef variants
        ("Good FAITH", "good faith"),
    ],
)
def test_search_normalization(raw: str, expected: str) -> None:
    assert normalize_for_search(raw) == expected


def test_embedding_retriever_with_injected_encoder(articles) -> None:
    def fake_encoder(texts):
        # tiny bag-of-keywords "embedding" so the test needs no model download
        keys = ["contract", "fault", "custom", "good faith"]
        return np.array([[float(k in t.lower()) for k in keys] + [0.01] for t in texts])

    r = EmbeddingRetriever(encoder=fake_encoder).fit(articles)
    assert r.query("fault damage", 1)[0].article["article_number"] == 163
