from __future__ import annotations

import pytest

from prodml.rag import LegalQAPredictor


def test_query_one_before_load_raises() -> None:
    predictor = LegalQAPredictor()
    with pytest.raises(RuntimeError):
        predictor.query_one("any question")


def test_query_one_returns_float_scores(trained_predictor: LegalQAPredictor) -> None:
    result = trained_predictor.query_one("ما هي شروط صحة العقد؟")
    assert isinstance(result["answer"], str)
    assert all(isinstance(s, float) for s in result["scores"])


def test_query_one_is_deterministic(trained_predictor: LegalQAPredictor) -> None:
    first = trained_predictor.query_one("العقد شريعة المتعاقدين")
    second = trained_predictor.query_one("العقد شريعة المتعاقدين")
    # trace_id is a fresh uuid per request by design; everything else must be identical
    assert {**first, "trace_id": ""} == {**second, "trace_id": ""}


def test_query_batch_matches_individual_queries(trained_predictor: LegalQAPredictor) -> None:
    questions = ["العقد شريعة المتعاقدين", "المسؤولية التقصيرية"]
    batch_results = trained_predictor.query_batch(questions)
    individual_results = [trained_predictor.query_one(q) for q in questions]
    strip = lambda rs: [{**r, "trace_id": ""} for r in rs]
    assert strip(batch_results) == strip(individual_results)


def test_metadata_reports_loaded_state(trained_predictor: LegalQAPredictor) -> None:
    meta = trained_predictor.metadata()
    assert meta["articles_indexed"] >= 5
    assert meta["loaded_at"] is not None


def test_unanswerable_question_does_not_crash(trained_predictor: LegalQAPredictor) -> None:
    result = trained_predictor.query_one("zzz gibberish qqq unrelated")
    assert "sources" in result


def test_out_of_scope_question_abstains(trained_predictor: LegalQAPredictor) -> None:
    result = trained_predictor.query_one("كيف أطبخ الكشري؟")
    assert result["abstained"] is True
    assert result["sources"] == [] and result["scores"] == []
    assert "not answering" in result["answer"]


def test_in_scope_question_cites_and_does_not_abstain(trained_predictor: LegalQAPredictor) -> None:
    result = trained_predictor.query_one("Must contracts be performed in good faith?")
    assert result["abstained"] is False
    assert result["sources"][0] == "Egyptian Civil Code, Article 148"
    assert result["confidence"] >= 0.25
    assert "not legal advice" in result["disclaimer"]
