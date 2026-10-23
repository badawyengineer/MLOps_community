"""The mock LLM must be clearly labelled and speak enough OpenAI protocol for the real client."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from prodml.llm import build_messages
from prodml.mock_llm import MOCK_PREFIX, create_app, make_answer

CONTEXT = "[Article 148] Contracts must be performed in good faith.\nنص\n\n[Article 147] Pacta sunt servanda.\nنص"


def _client() -> TestClient:
    return TestClient(create_app("test-model"))


def test_models_endpoint_lists_the_configured_id() -> None:
    assert _client().get("/v1/models").json()["data"][0]["id"] == "test-model"


def test_answer_is_labelled_and_cites_the_first_context_article() -> None:
    a = make_answer(build_messages("q?", CONTEXT)[1]["content"])
    assert a.startswith(MOCK_PREFIX) and "[Article 148]" in a and "[Article 147]" not in a


def test_no_context_yields_a_labelled_non_answer() -> None:
    assert make_answer("Question: hi").startswith(MOCK_PREFIX)


def test_chat_completion_reports_usage() -> None:
    body = {"model": "m", "messages": build_messages("q?", CONTEXT)}
    data = _client().post("/v1/chat/completions", json=body).json()
    assert data["choices"][0]["message"]["content"].startswith(MOCK_PREFIX)
    assert data["usage"]["completion_tokens"] > 0


def test_streaming_ends_with_usage_and_done() -> None:
    body = {
        "model": "m",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": build_messages("q?", CONTEXT),
    }
    lines = [
        ln
        for ln in _client().post("/v1/chat/completions", json=body).text.splitlines()
        if ln.startswith("data: ")
    ]
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[-2][6:])["usage"]["completion_tokens"] > 0
    text = "".join(json.loads(ln[6:])["choices"][0]["delta"]["content"] for ln in lines[:-2])
    assert text.startswith(MOCK_PREFIX)
