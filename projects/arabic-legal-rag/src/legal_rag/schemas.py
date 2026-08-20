from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class LegalQuestionRequest(BaseModel):
    question: Annotated[str, Field(min_length=3, max_length=2000)]
    language: Annotated[str, Field(default="ar", pattern="^(ar|en)$")]

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("question must not be empty")
        return value


class Citation(BaseModel):
    document: str
    page: int | None = None
    article: str | None = None
    source_type: str = "demo"


class LegalAnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model_version: str
    grounded: bool


class FeedbackRequest(BaseModel):
    request_id: Annotated[str, Field(min_length=1, max_length=100)]
    rating: Annotated[int, Field(ge=1, le=5)]
    comment: Annotated[str | None, Field(default=None, max_length=1000)] = None


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
