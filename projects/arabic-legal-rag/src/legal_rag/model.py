from .config import settings
from .schemas import Citation, LegalAnswerResponse


class LegalQAModel:
    """
    Session 1 placeholder for the future legal RAG pipeline.

    This deliberately does NOT claim to answer real legal questions.
    In later sessions this class can be replaced by:
        ingestion -> chunking -> embeddings -> vector DB -> retrieval
        -> prompt construction -> LLM/vLLM -> citations.
    """

    def __init__(self, version: str | None = None) -> None:
        self.version = version or settings.model_version

    def predict(self, question: str, language: str = "ar") -> LegalAnswerResponse:
        if language == "en":
            answer = (
                "Session 1 demo response. The real system will answer from "
                "indexed legal documents and provide source-grounded citations."
            )
        else:
            answer = (
                "هذه إجابة تجريبية للمرحلة الأولى فقط. النظام النهائي سيجيب "
                "اعتمادًا على المستندات القانونية المفهرسة مع إرفاق مصادر "
                "واستشهادات مرتبطة بالنص المسترجع."
            )

        return LegalAnswerResponse(
            answer=answer,
            citations=[
                Citation(
                    document="session1-demo-document",
                    page=1,
                    article=None,
                    source_type="demo",
                )
            ],
            model_version=self.version,
            grounded=False,
        )
