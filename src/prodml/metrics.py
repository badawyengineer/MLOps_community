"""Prometheus metrics. Beyond the usual request count/latency, we track the *retrieval
confidence distribution* and abstentions: a rising share of low-confidence or refused
queries is the earliest sign of drift (users asking things the corpus doesn't cover)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("prodml_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "prodml_request_latency_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
ABSTENTIONS = Counter("prodml_abstentions_total", "Questions refused for low retrieval confidence")
TOP_SCORE = Histogram(
    "prodml_top_score",
    "Top retrieval score per question (0 = nothing retrieved)",
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0),
)

__all__ = ["ABSTENTIONS", "CONTENT_TYPE_LATEST", "LATENCY", "REQUESTS", "TOP_SCORE", "render"]


def render() -> bytes:
    return generate_latest()


def observe_answer(result: dict) -> None:
    TOP_SCORE.observe(result.get("confidence", 0.0))
    if result.get("abstained"):
        ABSTENTIONS.inc()
