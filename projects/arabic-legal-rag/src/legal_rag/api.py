import time
import uuid

import structlog
from fastapi import FastAPI, Request

from .config import settings
from .logging_config import configure_logging, log
from .model import LegalQAModel
from .schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LegalAnswerResponse,
    LegalQuestionRequest,
)

configure_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Session 1 production skeleton for an Arabic Legal Document Q&A "
        "system. The current model is a demo placeholder; no real legal "
        "retrieval or legal advice is provided yet."
    ),
)

model = LegalQAModel()


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        endpoint=request.url.path,
        method=request.method,
    )

    started = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        log.info(
            "request.completed",
            status_code=status_code,
            latency_ms=latency_ms,
        )
        structlog.contextvars.clear_contextvars()


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
    )


@app.post("/predict", response_model=LegalAnswerResponse, tags=["legal-qa"])
async def predict(payload: LegalQuestionRequest) -> LegalAnswerResponse:
    log.info("legal_question.received", language=payload.language)
    return model.predict(payload.question, payload.language)


@app.post("/ask", response_model=LegalAnswerResponse, tags=["legal-qa"])
async def ask(payload: LegalQuestionRequest) -> LegalAnswerResponse:
    """
    Future production endpoint.

    In later sessions this endpoint will call the complete RAG pipeline.
    """
    log.info("legal_question.ask", language=payload.language)
    return model.predict(payload.question, payload.language)


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    log.info(
        "feedback.received",
        request_id=payload.request_id,
        rating=payload.rating,
        has_comment=payload.comment is not None,
    )
    return FeedbackResponse(status="received", request_id=payload.request_id)
