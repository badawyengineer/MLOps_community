from legal_rag.model import LegalQAModel


def test_model_returns_demo_response():
    model = LegalQAModel(version="test-v1")

    result = model.predict("ما هو القانون؟")

    assert result.model_version == "test-v1"
    assert result.grounded is False
    assert result.answer
    assert len(result.citations) == 1
    assert result.citations[0].document == "session1-demo-document"


def test_model_supports_english_request():
    model = LegalQAModel()

    result = model.predict("What is this system?", language="en")

    assert "Session 1 demo response" in result.answer


def test_model_uses_configured_version():
    model = LegalQAModel(version="custom-version")

    result = model.predict("اختبار")

    assert result.model_version == "custom-version"
