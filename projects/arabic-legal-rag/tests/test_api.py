from fastapi.testclient import TestClient

from legal_rag.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_success():
    response = client.post(
        "/predict",
        json={"question": "ما هو القانون المدني؟", "language": "ar"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "citations" in body
    assert body["grounded"] is False


def test_ask_success():
    response = client.post(
        "/ask",
        json={"question": "ما هي المسؤولية المدنية؟", "language": "ar"},
    )

    assert response.status_code == 200
    assert response.json()["model_version"]


def test_question_validation():
    response = client.post(
        "/ask",
        json={"question": "x", "language": "ar"},
    )

    assert response.status_code == 422


def test_language_validation():
    response = client.post(
        "/ask",
        json={"question": "ما هو القانون؟", "language": "fr"},
    )

    assert response.status_code == 422


def test_feedback_success():
    response = client.post(
        "/feedback",
        json={
            "request_id": "abc-123",
            "rating": 5,
            "comment": "Good demo response",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "received",
        "request_id": "abc-123",
    }


def test_feedback_rejects_invalid_rating():
    response = client.post(
        "/feedback",
        json={"request_id": "abc-123", "rating": 7},
    )

    assert response.status_code == 422
