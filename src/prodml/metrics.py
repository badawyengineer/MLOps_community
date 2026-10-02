"""Prometheus metrics. Beyond the usual request count/latency, we track the *retrieval
confidence distribution* and abstentions: a rising share of low-confidence or refused
queries is the earliest sign of drift (users asking things the corpus doesn't cover)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

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

LLM_TOKENS = Counter(
    "prodml_llm_tokens_total", "LLM tokens used", ["kind"]
)  # kind=prompt|completion
LLM_COST = Counter("prodml_llm_cost_usd_total", "Estimated LLM spend in USD (price from settings)")
LLM_FALLBACKS = Counter("prodml_llm_fallbacks_total", "LLM failures answered extractively instead")
GUARDRAIL_FLAGS = Counter("prodml_guardrail_flags_total", "Guardrail interventions", ["flag"])
FAITHFULNESS_PROXY = Histogram(
    "prodml_faithfulness_proxy",
    "Online lexical faithfulness proxy per answer (NOT RAGAS)",
    buckets=(0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
DRIFT_COSINE = Histogram(
    "prodml_query_centroid_cosine",
    "Cosine of each live query embedding to the eval-question centroid",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0),
)
DRIFT_DROP = Gauge(
    "prodml_query_drift_drop",
    "Baseline mean cosine minus rolling mean cosine (higher = more drift)",
)

__all__ = ["ABSTENTIONS", "CONTENT_TYPE_LATEST", "LATENCY", "REQUESTS", "TOP_SCORE", "render"]


def render() -> bytes:
    return generate_latest()


def observe_answer(result: dict) -> None:
    from prodml.config import settings

    TOP_SCORE.observe(result.get("confidence", 0.0))
    if result.get("abstained"):
        ABSTENTIONS.inc()
    usage = result.get("usage") or {}
    p, c = usage.get("prompt", 0), usage.get("completion", 0)
    if p or c:
        LLM_TOKENS.labels("prompt").inc(p)
        LLM_TOKENS.labels("completion").inc(c)
        LLM_COST.inc(
            p / 1000 * settings.llm_price_in_per_1k + c / 1000 * settings.llm_price_out_per_1k
        )
    for flag in result.get("guardrail_flags", []):
        GUARDRAIL_FLAGS.labels(flag).inc()
    if "llm_fallback" in result.get("guardrail_flags", []):
        LLM_FALLBACKS.inc()
    if result.get("faithfulness_proxy") is not None:
        FAITHFULNESS_PROXY.observe(result["faithfulness_proxy"])
