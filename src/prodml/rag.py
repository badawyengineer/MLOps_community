"""The RAG predictor — the one seam everything else sits behind.

Mirrors the handbook's guidance for Module 1 ("one class with a clear
interface... this is the seam that makes Modules 3, 4 and 5 possible"):
the API, the BentoML runner (Module 3), and a swapped-in vLLM generator
all sit behind ``LegalQAPredictor.load()`` / ``.query()`` /
``.query_batch()`` without those callers changing.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from prodml.config import settings
from prodml.corpus import Article, load_corpus, sample_corpus
from prodml.retriever import ScoredArticle, TfidfRetriever

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


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


class LegalQAPredictor:
    """Loads the article corpus + retriever once, then answers questions.

    Generation is currently *extractive*: the answer is composed from
    the top-matching article's text, cited by article number. This is a
    deliberate, honestly-labelled placeholder — Module 3 swaps
    ``_generate`` for a real call to a served LLM (vLLM or an API)
    without touching the retrieval logic or the API layer above it.
    """

    def __init__(self, corpus_path: Path | None = None, top_k: int | None = None) -> None:
        self._corpus_path = corpus_path or settings.corpus_path
        self._top_k = top_k or settings.top_k
        self._retriever: TfidfRetriever | None = None
        self._articles: list[Article] = []
        self._loaded_at: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._retriever is not None

    def load(self) -> LegalQAPredictor:
        """Load the corpus and fit the retriever. Called once at service
        startup via FastAPI's lifespan — never per-request."""
        if self._corpus_path.exists():
            self._articles = load_corpus(self._corpus_path)
        else:
            logger.warning(
                "Corpus file not found at %s — falling back to the sample " "bootstrap corpus.",
                self._corpus_path,
            )
            self._articles = sample_corpus()

        self._retriever = TfidfRetriever().fit(self._articles)
        self._loaded_at = time.time()
        logger.info("Loaded %d articles into the retriever.", len(self._articles))
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": settings.model_version,
            "articles_indexed": len(self._articles),
            "generation_backend": settings.generation_backend,
            "loaded_at": self._loaded_at,
        }

    @timed
    def query_one(self, question: str) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("LegalQAPredictor.load() must be called before query_one().")

        matches = self._retriever.query(question, top_k=self._top_k)
        answer = self._generate(question, matches)
        return {
            "answer": answer,
            "sources": [m.article["citation"] for m in matches],
            "scores": [round(m.score, 4) for m in matches],
        }

    def query_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        return [self.query_one(q) for q in questions]

    def _generate(self, question: str, matches: list[ScoredArticle]) -> str:
        """Extractive stand-in for a real generative call. Kept as a
        separate method precisely so Module 3 can replace only this
        method's body with a vLLM/API call."""
        if not matches:
            return (
                "No article in the indexed corpus matched this question "
                "closely enough to answer confidently."
            )
        top = matches[0].article
        return f"[{top['citation']}] {top['text_en']}"
