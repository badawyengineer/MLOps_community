from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prodml.corpus import Article, sample_corpus
from prodml.rag import LegalQAPredictor


@pytest.fixture()
def sample_articles() -> list[Article]:
    return sample_corpus()


@pytest.fixture(scope="session")
def trained_predictor() -> LegalQAPredictor:
    """Session-scoped so the (cheap) TF-IDF fit only happens once per
    test run — mirrors the handbook's fixture guidance."""
    predictor = LegalQAPredictor()
    predictor.load()
    return predictor


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, trained_predictor: LegalQAPredictor) -> TestClient:
    """API test client with the module-level predictor monkeypatched to
    an already-loaded one, so these tests don't depend on a real corpus
    file existing on disk at import time."""
    from prodml.api import main as api_main

    monkeypatch.setattr(api_main, "predictor", trained_predictor)
    with TestClient(api_main.app) as test_client:
        yield test_client
