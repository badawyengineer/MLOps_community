from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_200_when_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["documents_indexed"] >= 5


def test_metadata_returns_model_info(client: TestClient) -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert body["articles_indexed"] >= 5


def test_ask_happy_path(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "العقد شريعة المتعاقدين"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert isinstance(body["sources"], list)
    assert "correlation_id" in body
    assert body["latency_ms"] >= 0


def test_ask_echoes_correlation_id_header(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json={"question": "ما هي شروط صحة العقد؟"},
        headers={"X-Request-ID": "test-corr-id-123"},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-corr-id-123"
    assert response.json()["correlation_id"] == "test-corr-id-123"


def test_ask_rejects_empty_question_with_422(client: TestClient) -> None:
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_ask_rejects_missing_field_with_422(client: TestClient) -> None:
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_batch_happy_path(client: TestClient) -> None:
    response = client.post(
        "/ask/batch",
        json={"questions": ["العقد شريعة المتعاقدين", "المسؤولية التقصيرية"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2


def test_ask_batch_rejects_empty_list(client: TestClient) -> None:
    response = client.post("/ask/batch", json={"questions": []})
    assert response.status_code == 422


def test_ask_exposes_confidence_and_abstention(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "كيف أطبخ الكشري؟"}).json()
    assert body["abstained"] is True and body["sources"] == []
    ok = client.post("/ask", json={"question": "Must contracts be performed in good faith?"}).json()
    assert ok["abstained"] is False and ok["confidence"] > 0


def test_metrics_endpoint_reports_abstentions_and_requests(client: TestClient) -> None:
    client.post("/ask", json={"question": "كيف أطبخ الكشري؟"})
    text = client.get("/metrics").text
    assert "prodml_abstentions_total" in text
    assert 'prodml_requests_total{path="/ask",status="200"}' in text
    assert "prodml_top_score_bucket" in text


def test_metadata_reports_retriever_and_threshold(client: TestClient) -> None:
    meta = client.get("/metadata").json()
    assert meta["retriever"] == "hybrid-dense" and meta["min_confidence"] == 0.25


def test_ask_response_has_generation_and_guardrail_fields(client: TestClient) -> None:
    body = client.post(
        "/ask", json={"question": "Must contracts be performed in good faith?"}
    ).json()
    assert body["generation_backend"] == "extractive"
    assert body["guardrail_flags"] == [] and 0 <= body["faithfulness_proxy"] <= 1
    assert len(body["trace_id"]) == 32


def test_ask_stream_is_sse_with_meta_tokens_done(client: TestClient) -> None:
    import json as _json

    r = client.post("/ask/stream", json={"question": "Must contracts be performed in good faith?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [_json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[0]["type"] == "meta" and events[-1]["type"] == "done"
    assert sum(e["type"] == "token" for e in events) > 3  # progressive, not one blob
    assert events[-1]["sources"][0] == "Egyptian Civil Code, Article 148"


def test_ask_stream_validates_input(client: TestClient) -> None:
    assert client.post("/ask/stream", json={"question": "x"}).status_code == 422


def test_metrics_expose_new_monitoring_series(client: TestClient) -> None:
    client.post("/ask", json={"question": "Must contracts be performed in good faith?"})
    text = client.get("/metrics").text
    for series in (
        "prodml_faithfulness_proxy_bucket",
        "prodml_query_centroid_cosine_bucket",
        "prodml_query_drift_drop",
    ):
        assert series in text
