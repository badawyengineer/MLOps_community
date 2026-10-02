from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from prodml import metrics
from prodml.api.schemas import (
    AskBatchRequest,
    AskBatchResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    MetadataResponse,
)
from prodml.config import settings
from prodml.logging_conf import configure_logging, correlation_id_var, new_correlation_id
from prodml.rag import LegalQAPredictor

logger = logging.getLogger(__name__)

predictor = LegalQAPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model loaded once at startup — never inside a request handler.
    configure_logging(settings.log_level)
    predictor.load()
    yield


app = FastAPI(
    title="Arabic Legal Document Q&A",
    description="RAG service over the Egyptian Civil Code — ITI MLOps Practitioner final project (Track B).",
    version=settings.model_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Request-ID", new_correlation_id())
    token = correlation_id_var.set(correlation_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        correlation_id_var.reset(token)
    response.headers["X-Request-ID"] = correlation_id
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "path=%s status=%s latency_ms=%.2f", request.url.path, response.status_code, elapsed_ms
    )
    # Use the matched route template (not the raw URL) so label cardinality stays bounded.
    route = request.scope.get("route")
    path_label = getattr(route, "path", "unmatched")
    metrics.REQUESTS.labels(path_label, str(response.status_code)).inc()
    metrics.LATENCY.labels(path_label).observe(elapsed_ms / 1000)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("validation rejection: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # 200 only if the model is actually loaded in memory — not just "the process is alive".
    if not predictor.is_loaded:
        return JSONResponse(  # type: ignore[return-value]
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "loading", "documents_indexed": 0},
        )
    return HealthResponse(
        status="healthy", documents_indexed=predictor.metadata()["articles_indexed"]
    )


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE_LATEST)


@app.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    return MetadataResponse(**predictor.metadata())


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    start = time.perf_counter()
    result = predictor.query_one(payload.question)
    metrics.observe_answer(result)
    latency_ms = (time.perf_counter() - start) * 1000
    return AskResponse(
        **result,
        model_version=settings.model_version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/ask/batch", response_model=AskBatchResponse)
async def ask_batch(payload: AskBatchRequest) -> AskBatchResponse:
    start = time.perf_counter()
    raw_results = predictor.query_batch(payload.questions)
    for r in raw_results:
        metrics.observe_answer(r)
    latency_ms = (time.perf_counter() - start) * 1000
    per_item_latency = round(latency_ms / max(len(raw_results), 1), 2)
    results = [
        AskResponse(
            **r,
            model_version=settings.model_version,
            correlation_id=correlation_id_var.get(),
            latency_ms=per_item_latency,
        )
        for r in raw_results
    ]
    return AskBatchResponse(results=results)
