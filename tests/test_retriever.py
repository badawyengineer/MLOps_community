from __future__ import annotations

import pytest

from prodml.corpus import Article
from prodml.retriever import TfidfRetriever


@pytest.fixture()
def retriever(sample_articles: list[Article]) -> TfidfRetriever:
    return TfidfRetriever().fit(sample_articles)


def test_query_before_fit_raises() -> None:
    retriever = TfidfRetriever()
    with pytest.raises(RuntimeError):
        retriever.query("test")


def test_query_returns_at_most_top_k(retriever: TfidfRetriever) -> None:
    results = retriever.query("العقد شريعة المتعاقدين", top_k=2)
    assert len(results) <= 2


def test_query_is_deterministic_across_two_calls(retriever: TfidfRetriever) -> None:
    first = retriever.query("ما هي شروط صحة العقد؟", top_k=3)
    second = retriever.query("ما هي شروط صحة العقد؟", top_k=3)
    assert [r.article["article_number"] for r in first] == [
        r.article["article_number"] for r in second
    ]


def test_top_match_for_contract_question_is_article_147(retriever: TfidfRetriever) -> None:
    # "العقد شريعة المتعاقدين" is Article 147's own opening phrase — a
    # near-verbatim query should surface it as the top hit.
    results = retriever.query("العقد شريعة المتعاقدين وال يجوز نقضه", top_k=1)
    assert results, "expected at least one match"
    assert results[0].article["article_number"] == 147


def test_query_with_no_overlap_returns_no_results(retriever: TfidfRetriever) -> None:
    results = retriever.query("zzz completely unrelated gibberish qqq", top_k=3)
    assert results == [] or all(r.score < 0.3 for r in results)


@pytest.mark.parametrize("empty_query", ["", "   "])
def test_query_handles_empty_string_without_crashing(
    retriever: TfidfRetriever, empty_query: str
) -> None:
    results = retriever.query(empty_query, top_k=3)
    assert isinstance(results, list)
