"""The lexical-support guard: refuse when the top article shares (almost) no wording with the question."""

from __future__ import annotations

import pytest

from prodml.config import settings
from prodml.corpus import Article, load_corpus
from prodml.rag import LegalQAPredictor
from prodml.retriever import is_lexically_supported, lexical_support

ART_148: Article = {
    "article_number": 148,
    "text_en": "A contract must be performed in accordance with its contents and in compliance with good faith.",
    "text_ar": "يجب تنفيذ العقد طبقا لما اشتمل عليه وبطريقة تتفق مع ما توجبه حسن النية",
}


def test_related_question_is_supported() -> None:
    assert is_lexically_supported("Must contracts be performed in good faith?", ART_148, 2)


def test_unrelated_question_is_not_supported() -> None:
    assert not is_lexically_supported("How do I bake sourdough bread?", ART_148, 2)
    assert lexical_support("How do I bake sourdough bread?", ART_148)[0] == 0


def test_arabic_question_uses_arabic_text() -> None:
    assert is_lexically_supported("هل يجب تنفيذ العقد بحسن النية؟", ART_148, 2)
    assert not is_lexically_supported("كيف أطبخ الكشري؟", ART_148, 2)


def test_short_question_passes_when_half_its_words_match() -> None:
    # one content word, and it matches: shared/n = 1.0 >= 0.5 even though shared < 2
    assert is_lexically_supported("good faith?", ART_148, 2)


def test_zero_disables_the_check() -> None:
    assert is_lexically_supported("How do I bake sourdough bread?", ART_148, 0)


@pytest.fixture(scope="module")
def real_predictor() -> LegalQAPredictor:
    if len(load_corpus(settings.corpus_path)) < 1000:
        pytest.skip("needs the real 1,149-article corpus")
    p = LegalQAPredictor()
    p.load()
    return p


@pytest.mark.parametrize(
    "question",
    [
        "How do I bake sourdough bread?",
        "What is the penalty for drunk driving?",
        "كيف أطبخ الكشري؟",
    ],
)
def test_out_of_scope_questions_are_refused(
    real_predictor: LegalQAPredictor, question: str
) -> None:
    assert real_predictor.query_one(question)["abstained"] is True


def test_in_scope_question_is_still_answered(real_predictor: LegalQAPredictor) -> None:
    r = real_predictor.query_one("Must contracts be performed in good faith?")
    assert r["abstained"] is False and "Article 148" in r["sources"][0]


def test_guard_is_what_refuses_the_sourdough_question(
    real_predictor: LegalQAPredictor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the guard this question scores above the confidence threshold and is answered."""
    monkeypatch.setattr(settings, "min_lexical_overlap", 0)
    assert real_predictor.query_one("How do I bake sourdough bread?")["abstained"] is False
