"""LLM client, generation flow, streaming, guardrails, faithfulness, tracing, drift.
All HTTP is mocked with httpx.MockTransport - no network, no model."""

from __future__ import annotations

import json

import httpx
import pytest

from prodml.config import settings
from prodml.drift import EmbeddingDrift
from prodml.faithfulness import faithfulness_proxy
from prodml.guardrails import apply_guardrails, cited_articles, redact_pii
from prodml.llm import LLMError, OpenAICompatClient, build_messages
from prodml.rag import ABSTAIN_MESSAGE, LegalQAPredictor
from prodml.tracing import Tracer

GOOD_Q = "Must contracts be performed in good faith?"


def _chat_response(text: str, p: int = 50, c: int = 12) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": p, "completion_tokens": c},
        },
    )


def _sse(deltas: list[str], usage=(50, 9)) -> httpx.Response:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}" for d in deltas]
    lines.append(
        f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': usage[0], 'completion_tokens': usage[1]}})}"
    )
    lines.append("data: [DONE]")
    return httpx.Response(
        200, text="\n\n".join(lines), headers={"content-type": "text/event-stream"}
    )


def client_for(handler) -> OpenAICompatClient:
    return OpenAICompatClient(
        "http://llm/v1", "test-model", "k", transport=httpx.MockTransport(handler)
    )


def predictor_with(handler) -> LegalQAPredictor:
    return LegalQAPredictor(llm=client_for(handler)).load()


# ---------------- LLM client ----------------
def test_complete_parses_text_and_usage_and_sends_auth() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return _chat_response("hello [Article 148]")

    res = client_for(handler).complete(build_messages("q", "ctx"))
    assert res.text == "hello [Article 148]" and (res.prompt_tokens, res.completion_tokens) == (
        50,
        12,
    )
    assert (
        seen["auth"] == "Bearer k"
        and seen["body"]["temperature"] == 0
        and seen["body"]["model"] == "test-model"
    )


def test_stream_yields_deltas_and_records_usage() -> None:
    c = client_for(lambda r: _sse(["Good ", "faith ", "[Article 148]"]))
    assert list(c.stream(build_messages("q", "ctx"))) == ["Good ", "faith ", "[Article 148]"]
    assert (c.last_usage.prompt_tokens, c.last_usage.completion_tokens) == (50, 9)


@pytest.mark.parametrize(
    "response", [httpx.Response(500, text="boom"), httpx.Response(200, json={"nope": 1})]
)
def test_complete_raises_llmerror_on_bad_responses(response) -> None:
    with pytest.raises(LLMError):
        client_for(lambda r: response).complete([])


def test_stream_raises_llmerror_on_http_failure() -> None:
    with pytest.raises(LLMError):
        list(client_for(lambda r: httpx.Response(503)).stream([]))


# ---------------- generation flow ----------------
def test_llm_answer_used_and_tokens_reported() -> None:
    p = predictor_with(
        lambda r: _chat_response("Contracts must be performed in good faith [Article 148].")
    )
    out = p.query_one(GOOD_Q)
    assert out["answer"].startswith("Contracts must be performed in good faith")
    assert out["usage"] == {"prompt": 50, "completion": 12} and out["guardrail_flags"] == []
    assert out["faithfulness_proxy"] is not None and 0 <= out["faithfulness_proxy"] <= 1


def test_prompt_contains_only_retrieved_articles_and_question() -> None:
    captured = {}

    def handler(req):
        captured["messages"] = json.loads(req.content)["messages"]
        return _chat_response("ok [Article 148]")

    predictor_with(handler).query_one(GOOD_Q)
    user_msg = captured["messages"][1]["content"]
    assert "[Article 148]" in user_msg and GOOD_Q in user_msg
    assert "Answer ONLY from the numbered articles" in captured["messages"][0]["content"]


def test_llm_outage_falls_back_to_extractive_and_is_flagged() -> None:
    out = predictor_with(lambda r: httpx.Response(500)).query_one(GOOD_Q)
    assert "llm_fallback" in out["guardrail_flags"]
    assert out["answer"].startswith("[Egyptian Civil Code, Article 148]")


def test_hallucinated_citation_is_replaced_with_grounded_text() -> None:
    out = predictor_with(lambda r: _chat_response("See [Article 999] for details.")).query_one(
        GOOD_Q
    )
    assert "unsupported_citation" in out["guardrail_flags"]
    assert "999" not in out["answer"] and "Article 148" in out["answer"]


def test_pii_in_llm_output_is_redacted() -> None:
    out = predictor_with(
        lambda r: _chat_response("Call 01012345678 or mail a@b.com about [Article 148].")
    ).query_one(GOOD_Q)
    assert "01012345678" not in out["answer"] and "a@b.com" not in out["answer"]
    assert {"pii:phone", "pii:email"} <= set(out["guardrail_flags"])


def test_abstention_never_calls_the_llm() -> None:
    calls = []
    p = predictor_with(lambda r: calls.append(1) or _chat_response("x"))
    out = p.query_one("كيف أطبخ الكشري؟")
    assert out["abstained"] and out["answer"] == ABSTAIN_MESSAGE and calls == []


def test_missing_llm_model_name_fails_fast(monkeypatch) -> None:
    monkeypatch.setattr(settings, "generation_backend", "vllm")
    monkeypatch.setattr(settings, "llm_model", "")
    with pytest.raises(ValueError, match="PRODML_LLM_MODEL"):
        LegalQAPredictor().load()


# ---------------- streaming ----------------
def test_stream_events_order_and_content_with_llm() -> None:
    p = predictor_with(lambda r: _sse(["Good ", "faith ", "[Article 148]."]))
    events = list(p.stream_events(GOOD_Q))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and kinds.count("token") == 3
    assert events[0]["sources"][0] == "Egyptian Civil Code, Article 148"  # top-k, best first
    assert events[-1]["usage"] == {"prompt": 50, "completion": 9}


def test_stream_replace_event_when_guardrail_changes_answer() -> None:
    p = predictor_with(lambda r: _sse(["Per ", "[Article 777]."]))
    events = list(p.stream_events(GOOD_Q))
    replace = [e for e in events if e["type"] == "replace"]
    assert (
        replace
        and "777" not in replace[0]["text"]
        and "unsupported_citation" in events[-1]["guardrail_flags"]
    )


def test_stream_extractive_and_abstain_paths() -> None:
    p = LegalQAPredictor().load()
    ev = list(p.stream_events(GOOD_Q))
    assert (
        "".join(e["text"] for e in ev if e["type"] == "token")
        .strip()
        .startswith("[Egyptian Civil Code, Article 148]")
    )
    ab = list(p.stream_events("كيف أطبخ الكشري؟"))
    assert ab[0]["abstained"] and ab[-1]["abstained"]


def test_stream_llm_failure_falls_back() -> None:
    ev = list(predictor_with(lambda r: httpx.Response(500)).stream_events(GOOD_Q))
    assert "llm_fallback" in ev[-1]["guardrail_flags"]


# ---------------- guardrails ----------------
@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail me at x.y@z.org", "email"),
        ("رقمي ٠١٠١٢٣٤٥٦٧٨", "phone"),
        ("+20 1012345678", "phone"),
        ("id 29801011234567 ok", "national_id"),
        ("card 4111 1111 1111 1111", "card"),
    ],
)
def test_pii_patterns(text, kind) -> None:
    out, kinds = redact_pii(text)
    assert kind in kinds and "REDACTED" in out


def test_clean_text_is_untouched() -> None:
    text = "Article 147 and 148 apply, ٣ مواد"
    assert redact_pii(text) == (text, [])


def test_cited_articles_handles_both_languages_and_digits() -> None:
    assert cited_articles("[Article 148] and المادة ١٤٧ and مادة 163") == {148, 147, 163}
    assert apply_guardrails("per [Article 1]", {1}).flags == []


# ---------------- faithfulness proxy ----------------
def test_proxy_high_when_supported_low_when_invented() -> None:
    ctx = ["A contract must be performed in accordance with its contents and good faith."]
    assert faithfulness_proxy("A contract must be performed in good faith.", ctx) == 1.0
    assert faithfulness_proxy("Penalties include imprisonment for ten years.", ctx) == 0.0
    mixed = faithfulness_proxy(
        "A contract must be performed in good faith. Imprisonment applies to breach.", ctx
    )
    assert mixed == 0.5
    assert faithfulness_proxy("", ctx) == 0.0


# ---------------- tracing ----------------
def test_trace_exports_langfuse_batch_with_score_and_usage() -> None:
    sent = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append((req.url.path, req.headers["authorization"][:5], json.loads(req.content)))
        return httpx.Response(207, json={})

    tracer = Tracer("http://lf:3000", "pk", "sk", transport=httpx.MockTransport(handler))
    p = LegalQAPredictor(
        llm=client_for(lambda r: _chat_response("ok [Article 148].")), tracer=tracer
    ).load()
    out = p.query_one(GOOD_Q)
    path, auth, body = sent[0]
    types = [e["type"] for e in body["batch"]]
    assert path == "/api/public/ingestion" and auth == "Basic"
    assert {"trace-create", "span-create", "generation-create", "score-create"} <= set(types)
    gen = next(e for e in body["batch"] if e["type"] == "generation-create")["body"]
    assert gen["usage"]["input"] == 50 and gen["traceId"] == out["trace_id"]


def test_tracing_is_noop_without_keys_and_never_raises_on_export_failure() -> None:
    assert not Tracer().enabled
    broken = Tracer(
        "http://lf", "pk", "sk", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    broken.start("x", "q").finish("a")  # must not raise


# ---------------- drift ----------------
def test_drift_detects_off_distribution_queries() -> None:
    base = [
        "هل يجوز تعديل العقد",
        "ما حكم الخطأ الذي يسبب ضررا للغير",
        "must contracts be performed in good faith",
    ]
    d = EmbeddingDrift(base, window=5)
    for q in base:
        d.observe(q)
    in_dist = d.drop_from_baseline
    for q in ["xkcd quantum zebra pizza", "اسعار الذهب اليوم", "football score tonight"] * 2:
        d.observe(q)
    assert d.drop_from_baseline > in_dist + 0.05
