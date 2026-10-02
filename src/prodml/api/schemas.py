from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="A question in Arabic or English about the indexed legal corpus.",
    )

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"question": "ما هي شروط صحة العقد؟"}]}
    )


class AskBatchRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=50)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    scores: list[float]
    abstained: bool = False
    confidence: float = 0.0
    disclaimer: str = ""
    model_version: str
    correlation_id: str
    latency_ms: float


class AskBatchResponse(BaseModel):
    results: list[AskResponse]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


class MetadataResponse(BaseModel):
    model_version: str
    articles_indexed: int
    generation_backend: str
    retriever: str = "tfidf"
    min_confidence: float = 0.0
    loaded_at: float | None
