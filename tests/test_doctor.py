from __future__ import annotations

from prodml import doctor
from prodml.config import settings


def _levels(results):
    return {name: level for level, name, _ in results}


def test_default_offline_setup_has_no_blocking_problem() -> None:
    results = doctor.run_checks(online=False)
    assert doctor.ERROR not in {lvl for lvl, _, _ in results}
    assert _levels(results)["LLM endpoint"] == doctor.SKIP  # extractive: nothing to probe


def test_missing_corpus_is_a_blocking_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "corpus_path", tmp_path / "missing.json")
    assert _levels(doctor.run_checks())["corpus"] == doctor.ERROR


def test_unreachable_llm_is_an_error_only_when_probing(monkeypatch) -> None:
    monkeypatch.setattr(settings, "generation_backend", "api")
    monkeypatch.setattr(settings, "llm_model", "m")
    monkeypatch.setattr(
        settings, "llm_base_url", "http://127.0.0.1:9/v1"
    )  # discard port: nothing listens
    monkeypatch.setattr(settings, "llm_api_key", "SUPERSECRET-key")
    offline = doctor.run_checks(online=False)
    online = doctor.run_checks(online=True)
    assert _levels(offline)["LLM endpoint"] == doctor.SKIP
    assert _levels(online)["LLM endpoint"] == doctor.ERROR
    assert "SUPERSECRET" not in str(offline) + str(online)
