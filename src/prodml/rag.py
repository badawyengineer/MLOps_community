"""The RAG predictor — the one seam everything else sits behind.

Mirrors the handbook's guidance for Module 1 ("one class with a clear
interface... this is the seam that makes Modules 3, 4 and 5 possible"):
the API, the BentoML runner (Module 3), and a swapped-in vLLM generator
all sit behind ``LegalQAPredictor.load()`` / ``.query()`` /
``.query_batch()`` without those callers changing.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from prodml import metrics
from prodml.config import settings
from prodml.corpus import Article, load_corpus, sample_corpus
from prodml.drift import EmbeddingDrift
from prodml.faithfulness import faithfulness_proxy
from prodml.guardrails import apply_guardrails
from prodml.llm import LLMError, OpenAICompatClient, build_messages
from prodml.retriever import Retriever, ScoredArticle, build_retriever
from prodml.tracing import Tracer, now

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

ABSTAIN_MESSAGE = (
    "I could not find an article in the indexed corpus that addresses this question "
    "with enough confidence, so I am not answering. Please rephrase, or consult a lawyer."
)
DISCLAIMER = "Informational retrieval over statute text only; not legal advice."


def timed(func: F) -> F:
    """Decorator logging execution time of the function it wraps.
    Written by hand (per the Module 1 requirement) rather than pulled
    from a library — this is exactly what Module 4's benchmark harness
    builds on top of.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("%s took %.2fms", func.__qualname__, elapsed_ms)
        return result

    return wrapper  # type: ignore[return-value]


@dataclass
class Retrieval:
    matches: list[ScoredArticle]
    confidence: float
    abstained: bool


def _build_context(matches: list[ScoredArticle], max_chars: int) -> str:
    parts, used = [], 0
    for m in matches:
        a = m.article
        block = f"[Article {a['article_number']}] {a['text_en']}\n{a['text_ar']}".strip()
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _extractive(matches: list[ScoredArticle]) -> str:
    if not matches:
        return "No article in the indexed corpus matched this question closely enough to answer confidently."
    top = matches[0].article
    return f"[{top['citation']}] {top['text_en']}"


class LegalQAPredictor:
    """Loads corpus + retriever (+ optional LLM) once, then answers questions.

    Retrieval -> abstain-or-generate -> guardrails -> faithfulness proxy, with a trace per
    request. ``generation_backend`` selects the generator: ``extractive`` (default, offline,
    returns the top article's text) or ``vllm`` / ``api`` (any OpenAI-compatible endpoint;
    on failure we fall back to extractive and flag it - never a 500 for a flaky LLM).
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
        top_k: int | None = None,
        llm: OpenAICompatClient | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._corpus_path = corpus_path or settings.corpus_path
        self._top_k = top_k or settings.top_k
        self._retriever: Retriever | None = None
        self._articles: list[Article] = []
        self._loaded_at: float | None = None
        self._llm = llm
        self._tracer = tracer or Tracer(
            settings.langfuse_host, settings.langfuse_public_key, settings.langfuse_secret_key
        )
        self._drift: EmbeddingDrift | None = None

    @property
    def is_loaded(self) -> bool:
        return self._retriever is not None

    def load(self) -> LegalQAPredictor:
        """Called once at service startup via FastAPI's lifespan - never per-request."""
        if self._corpus_path.exists():
            self._articles = load_corpus(self._corpus_path)
        else:
            logger.warning(
                "Corpus file not found at %s - using the sample bootstrap corpus.",
                self._corpus_path,
            )
            self._articles = sample_corpus()

        retriever = build_retriever(settings.retriever)
        if hasattr(retriever, "load"):  # dense retrievers: prefer the persisted (image-baked) index
            try:
                retriever.load(settings.index_dir, self._articles)
                logger.info("Loaded persisted vector index from %s", settings.index_dir)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                logger.warning("No usable persisted index (%s); fitting in memory.", exc)
                retriever.fit(self._articles)
        else:
            retriever.fit(self._articles)
        self._retriever = retriever

        if self._llm is None and settings.generation_backend in ("vllm", "api"):
            if not settings.llm_model:
                raise ValueError(
                    "PRODML_LLM_MODEL must be set when generation_backend is 'vllm' or 'api'."
                )
            self._llm = OpenAICompatClient(
                settings.llm_base_url,
                settings.llm_model,
                settings.llm_api_key,
                settings.llm_timeout,
                settings.llm_max_tokens,
            )
        self._load_drift_baseline()
        self._loaded_at = time.time()
        logger.info(
            "Loaded %d articles; retriever=%s generation=%s",
            len(self._articles),
            settings.retriever,
            settings.generation_backend,
        )
        return self

    def _load_drift_baseline(self) -> None:
        path = settings.drift_baseline_path
        if not path.exists():
            return
        try:
            items = json.loads(path.read_text(encoding="utf-8"))["items"]
            qs = [i["question"] for i in items if i.get("answerable")]
            if qs:
                self._drift = EmbeddingDrift(qs)
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("drift baseline unavailable: %s", exc)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": settings.model_version,
            "articles_indexed": len(self._articles),
            "generation_backend": settings.generation_backend,
            "retriever": settings.retriever,
            "min_confidence": settings.min_confidence,
            "loaded_at": self._loaded_at,
        }

    # ---- shared steps -------------------------------------------------------------------
    def _retrieve(self, question: str) -> Retrieval:
        if not self.is_loaded:
            raise RuntimeError("LegalQAPredictor.load() must be called before querying.")
        if self._drift is not None:
            metrics.DRIFT_COSINE.observe(max(0.0, self._drift.observe(question)))
            metrics.DRIFT_DROP.set(self._drift.drop_from_baseline)
        matches = self._retriever.query(question, top_k=self._top_k)
        confidence = round(matches[0].score, 4) if matches else 0.0
        abstained = not matches or confidence < settings.min_confidence
        return Retrieval([] if abstained else matches, confidence, abstained)

    def _finish(
        self, text: str, matches: list[ScoredArticle], flags: list[str]
    ) -> tuple[str, list[str], float]:
        """Guardrails + faithfulness proxy on a finished answer."""
        guard = apply_guardrails(text, {m.article["article_number"] for m in matches})
        flags = flags + guard.flags
        if guard.unsupported_citations:
            logger.warning(
                "answer cited articles outside the retrieved set: %s", guard.unsupported_citations
            )
            guard.text = _extractive(matches)
        contexts = [f"{m.article['text_en']} {m.article['text_ar']}" for m in matches]
        return guard.text, flags, round(faithfulness_proxy(guard.text, contexts), 4)

    def _generate_full(
        self, question: str, matches: list[ScoredArticle]
    ) -> tuple[str, dict[str, int], list[str]]:
        if self._llm is None:
            return _extractive(matches), {}, []
        try:
            res = self._llm.complete(
                build_messages(question, _build_context(matches, settings.max_context_chars))
            )
            return res.text, {"prompt": res.prompt_tokens, "completion": res.completion_tokens}, []
        except LLMError as exc:
            logger.error("%s - falling back to extractive answer", exc)
            return _extractive(matches), {}, ["llm_fallback"]

    # ---- public API ---------------------------------------------------------------------
    @timed
    def query_one(self, question: str) -> dict[str, Any]:
        trace = self._tracer.start("ask", question)
        t0 = now()
        r = self._retrieve(question)
        trace.span(
            "retrieve", t0, now(), confidence=r.confidence, n=len(r.matches), abstained=r.abstained
        )

        usage: dict[str, int] = {}
        flags: list[str] = []
        proxy: float | None = None
        if r.abstained:
            answer = ABSTAIN_MESSAGE
        else:
            raw, usage, flags = self._generate_full(question, r.matches)
            if self._llm is not None and "llm_fallback" not in flags:
                trace.generation(
                    "answer",
                    settings.llm_model,
                    raw,
                    usage.get("prompt", 0),
                    usage.get("completion", 0),
                )
            answer, flags, proxy = self._finish(raw, r.matches, flags)
            trace.score("faithfulness_proxy", proxy)
        trace.finish(answer)
        return self._result(answer, r, flags, proxy, usage, trace.id)

    def stream_events(self, question: str) -> Iterator[dict[str, Any]]:
        """Server-sent-event payloads: ``meta`` -> ``token``* -> optional ``replace`` -> ``done``.

        Tokens are sent as they are generated. Guardrails need the whole answer, so if they
        change it (PII redaction, unsupported citation) a final ``replace`` event carries the
        safe text and clients MUST display it in place of the streamed tokens.
        """
        trace = self._tracer.start("ask_stream", question)
        r = self._retrieve(question)
        yield {
            "type": "meta",
            "sources": [m.article["citation"] for m in r.matches],
            "confidence": r.confidence,
            "abstained": r.abstained,
        }
        usage: dict[str, int] = {}
        flags: list[str] = []
        proxy: float | None = None
        if r.abstained:
            answer = ABSTAIN_MESSAGE
            yield {"type": "token", "text": answer}
        else:
            parts: list[str] = []
            if self._llm is not None:
                try:
                    for delta in self._llm.stream(
                        build_messages(
                            question, _build_context(r.matches, settings.max_context_chars)
                        )
                    ):
                        parts.append(delta)
                        yield {"type": "token", "text": delta}
                    u = self._llm.last_usage
                    usage = {"prompt": u.prompt_tokens, "completion": u.completion_tokens}
                except LLMError as exc:
                    logger.error("%s - falling back to extractive answer", exc)
                    flags.append("llm_fallback")
                    parts = [_extractive(r.matches)]
                    yield {"type": "token", "text": parts[0]}
            else:
                for word in _extractive(r.matches).split(" "):
                    parts.append(word + " ")
                    yield {"type": "token", "text": word + " "}
            streamed = "".join(parts)
            answer, flags, proxy = self._finish(streamed.strip(), r.matches, flags)
            trace.score("faithfulness_proxy", proxy)
            if answer != streamed.strip():
                yield {"type": "replace", "text": answer}
        trace.finish(answer)
        yield {"type": "done", **self._result(answer, r, flags, proxy, usage, trace.id)}

    def query_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        return [self.query_one(q) for q in questions]

    @staticmethod
    def _result(
        answer: str,
        r: Retrieval,
        flags: list[str],
        proxy: float | None,
        usage: dict[str, int],
        trace_id: str,
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "sources": [m.article["citation"] for m in r.matches],
            "scores": [round(m.score, 4) for m in r.matches],
            "abstained": r.abstained,
            "confidence": r.confidence,
            "disclaimer": DISCLAIMER,
            "generation_backend": settings.generation_backend,
            "guardrail_flags": flags,
            "faithfulness_proxy": proxy,
            "usage": usage,
            "trace_id": trace_id,
        }
