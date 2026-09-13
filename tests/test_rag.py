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
    assert first == second


def test_query_batch_matches_individual_queries(trained_predictor: LegalQAPredictor) -> None:
    questions = ["العقد شريعة المتعاقدين", "المسؤولية التقصيرية"]
    batch_results = trained_predictor.query_batch(questions)
    individual_results = [trained_predictor.query_one(q) for q in questions]
    assert batch_results == individual_results


def test_metadata_reports_loaded_state(trained_predictor: LegalQAPredictor) -> None:
    meta = trained_predictor.metadata()
    assert meta["articles_indexed"] >= 5
    assert meta["loaded_at"] is not None


def test_unanswerable_question_does_not_crash(trained_predictor: LegalQAPredictor) -> None:
    result = trained_predictor.query_one("zzz gibberish qqq unrelated")
    assert "sources" in result
