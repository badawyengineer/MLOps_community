"""Retrieval layer.

Four interchangeable retrievers behind one tiny interface (``fit`` / ``query``),
selected by ``PRODML_RETRIEVER``:

  * ``tfidf``     - char n-gram TF-IDF + cosine (the Module 1 baseline).
  * ``bm25``      - word-level BM25 with light Arabic stemming and stopwords.
  * ``hybrid``    - weighted blend of the two; scores stay in [0, 1] so a
                    single confidence threshold can drive abstention.
  * ``embedding`` - multilingual sentence embeddings (optional extra,
                    ``pip install -e ".[embeddings]"``).

All scores are normalized to roughly [0, 1] on purpose: the predictor abstains
below a confidence threshold, which only works if scores mean the same thing
across retrievers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from prodml.chunking import chunk_corpus, chunk_index_texts
from prodml.corpus import Article, normalize_for_search
from prodml.embedders import Embedder, build_embedder
from prodml.vectorstore import VectorStore


@dataclass
class ScoredArticle:
    article: Article
    score: float


class Retriever(Protocol):
    def fit(self, articles: list[Article]) -> Retriever: ...

    def query(self, question: str, top_k: int = 3) -> list[ScoredArticle]: ...


def _index_text(a: Article) -> str:
    return normalize_for_search(f"{a['topic']} {a['text_ar']} {a['text_en']}")


def _top_k(articles: list[Article], scores: np.ndarray, top_k: int) -> list[ScoredArticle]:
    ranked = scores.argsort()[::-1][:top_k]
    return [
        ScoredArticle(article=articles[i], score=float(scores[i])) for i in ranked if scores[i] > 0
    ]


class TfidfRetriever:
    """Char n-gram TF-IDF index over normalized article text."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._matrix = None
        self._articles: list[Article] = []

    def fit(self, articles: list[Article]) -> TfidfRetriever:
        self._articles = articles
        self._matrix = self._vectorizer.fit_transform([_index_text(a) for a in articles])
        return self

    def query(self, question: str, top_k: int = 3) -> list[ScoredArticle]:
        if self._matrix is None:
            raise RuntimeError("TfidfRetriever.fit() must be called before query().")
        query_vec = self._vectorizer.transform([normalize_for_search(question)])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        return _top_k(self._articles, scores, top_k)


# Question words / function words that carry no legal meaning. Deliberately short:
# a long stopword list on a small legal corpus risks deleting real signal.
_AR_STOPWORDS = frozenset(
    normalize_for_search(w)
    for w in [
        "ما",
        "هي",
        "هو",
        "هل",
        "في",
        "من",
        "الى",
        "علي",
        "عن",
        "مع",
        "ان",
        "او",
        "ثم",
        "كل",
        "اي",
        "كيف",
        "متي",
        "لماذا",
        "هذا",
        "هذه",
        "ذلك",
        "التي",
        "الذي",
        "يجوز",
        "يكون",
        "تكون",
        "كان",
    ]
)
_EN_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "be",
        "what",
        "how",
        "when",
        "why",
        "which",
        "who",
        "whom",
        "does",
        "do",
        "can",
        "may",
        "shall",
        "by",
        "with",
        "as",
        "at",
        "it",
        "its",
        "that",
        "this",
        "these",
        "those",
        "any",
    ]
)
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def tokenize(text: str) -> list[str]:
    """Normalize, split, drop stopwords, and lightly stem Arabic definite-article prefixes."""
    tokens = []
    for tok in _TOKEN_RE.findall(normalize_for_search(text)):
        if tok in _AR_STOPWORDS or tok in _EN_STOPWORDS:
            continue
        for prefix in _AR_PREFIXES:
            if tok.startswith(prefix) and len(tok) - len(prefix) >= 3:
                tok = tok[len(prefix) :]
                break
        tokens.append(tok)
    return tokens


class Bm25Retriever:
    """Okapi BM25 over word tokens. Implemented directly (no extra dependency).

    Scores are divided by the best score a query could possibly earn on this
    index (every query term matching with saturated term frequency), which puts
    them in [0, 1] and makes them comparable to cosine similarity.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._articles: list[Article] = []
        self._tf: list[Counter[str]] = []
        self._lengths = np.zeros(0)
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def fit(self, articles: list[Article]) -> Bm25Retriever:
        self._articles = articles
        docs = [tokenize(f"{a['topic']} {a['text_ar']} {a['text_en']}") for a in articles]
        self._tf = [Counter(d) for d in docs]
        self._lengths = np.array([len(d) for d in docs], dtype=float)
        self._avgdl = float(self._lengths.mean()) if len(docs) else 0.0
        n = len(docs)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        return self

    def query(self, question: str, top_k: int = 3) -> list[ScoredArticle]:
        if not self._articles:
            raise RuntimeError("Bm25Retriever.fit() must be called before query().")
        terms = [t for t in set(tokenize(question)) if t in self._idf]
        if not terms:
            return []
        scores = np.zeros(len(self._articles))
        for i, tf in enumerate(self._tf):
            norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / (self._avgdl or 1.0))
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    scores[i] += self._idf[t] * f * (self.k1 + 1) / (f + norm)
        ceiling = sum(self._idf[t] for t in terms) * (self.k1 + 1)
        return _top_k(self._articles, np.clip(scores / ceiling, 0.0, 1.0), top_k)


class HybridRetriever:
    """Weighted blend of char-TF-IDF (robust to spelling/morphology) and BM25
    (robust to rare, discriminating words). ``alpha`` weights the TF-IDF side."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._tfidf, self._bm25 = TfidfRetriever(), Bm25Retriever()
        self._articles: list[Article] = []

    def fit(self, articles: list[Article]) -> HybridRetriever:
        self._articles = articles
        self._tfidf.fit(articles)
        self._bm25.fit(articles)
        return self

    def query(self, question: str, top_k: int = 3) -> list[ScoredArticle]:
        n = len(self._articles)
        by_id: dict[int, float] = {}
        for weight, retr in ((self.alpha, self._tfidf), (1 - self.alpha, self._bm25)):
            for m in retr.query(question, top_k=n):
                key = m.article["article_number"]
                by_id[key] = by_id.get(key, 0.0) + weight * m.score
        lookup = {a["article_number"]: a for a in self._articles}
        ranked = sorted(by_id.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [ScoredArticle(article=lookup[k], score=s) for k, s in ranked if s > 0]


class EmbeddingRetriever:
    """Dense retrieval with a multilingual sentence-embedding model.

    ``encoder`` maps ``list[str] -> 2D array``; it is injectable so the logic is
    unit-testable without downloading a model. By default it lazily loads a
    ``sentence-transformers`` model (the ``embeddings`` extra).
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-small"

    def __init__(
        self,
        model_name: str | None = None,
        encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
    ) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._encoder = encoder
        self._articles: list[Article] = []
        self._matrix = np.zeros((0, 0))

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    'Embedding retrieval needs the extra: pip install -e ".[embeddings]"'
                ) from exc
            model = SentenceTransformer(self._model_name)
            self._encoder = lambda t: model.encode(list(t), normalize_embeddings=True)
        vecs = np.asarray(self._encoder(texts), dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def fit(self, articles: list[Article]) -> EmbeddingRetriever:
        self._articles = articles
        self._matrix = self._encode([_index_text(a) for a in articles])
        return self

    def query(self, question: str, top_k: int = 3) -> list[ScoredArticle]:
        if not self._articles:
            raise RuntimeError("EmbeddingRetriever.fit() must be called before query().")
        q = self._encode([normalize_for_search(question)])[0]
        return _top_k(self._articles, np.clip(self._matrix @ q, 0.0, 1.0), top_k)


def corpus_fingerprint(articles: list[Article]) -> str:
    import hashlib
    import json

    blob = json.dumps(articles, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class VectorRetriever:
    """Chunk by article -> embed -> vector store -> cosine search, aggregated to ARTICLES
    (an article's score is its best chunk's score, so results always cite article numbers)."""

    def __init__(
        self, embedder: Embedder | None = None, chunk_chars: int = 0, overlap: int = 0
    ) -> None:
        self.embedder = embedder or build_embedder("hashing")
        self.chunk_chars, self.overlap = chunk_chars, overlap
        self._articles: list[Article] = []
        self._by_number: dict[int, Article] = {}
        self.store: VectorStore | None = None

    def fit(self, articles: list[Article]) -> VectorRetriever:
        self._set_articles(articles)
        chunks = chunk_corpus(articles, self.chunk_chars, self.overlap)
        texts = chunk_index_texts(chunks)
        self.embedder.fit(texts)
        payloads = [{"chunk_id": c.chunk_id, "article_number": c.article_number} for c in chunks]
        self.store = VectorStore(
            self.embedder.encode(texts), payloads, self._meta(articles, len(chunks))
        )
        return self

    def _set_articles(self, articles: list[Article]) -> None:
        self._articles = articles
        self._by_number = {a["article_number"]: a for a in articles}

    def _meta(self, articles: list[Article], n_chunks: int) -> dict:
        return {
            "embedding_model": self.embedder.name,
            "chunk_chars": self.chunk_chars,
            "overlap": self.overlap,
            "n_articles": len(articles),
            "n_chunks": n_chunks,
            "corpus_fingerprint": corpus_fingerprint(articles),
        }

    def save(self, directory) -> None:
        if not self.embedder.stateless:
            raise ValueError(
                f"{self.embedder.name} is fitted per corpus and cannot be persisted; use a stateless embedder"
            )
        assert self.store is not None
        self.store.save(directory)

    def load(self, directory, articles: list[Article]) -> VectorRetriever:
        """Load a persisted index; raises ValueError if it is stale (different corpus/config)."""
        from pathlib import Path

        store = VectorStore.load(Path(directory))
        expect = self._meta(articles, store.meta.get("n_chunks", -1))
        if store.meta != expect:
            raise ValueError(f"index at {directory} is stale or built with a different config")
        self._set_articles(articles)
        self.store = store
        return self

    def query(self, question: str, top_k: int = 3) -> list[ScoredArticle]:
        if self.store is None:
            raise RuntimeError("VectorRetriever.fit() or load() must be called before query().")
        q = self.embedder.encode([normalize_for_search(question)])[0]
        best: dict[int, float] = {}
        for payload, score in self.store.search(q, k=max(top_k * 5, 20)):
            n = payload["article_number"]
            best[n] = max(best.get(n, 0.0), score)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            ScoredArticle(self._by_number[n], float(np.clip(sc, 0.0, 1.0)))
            for n, sc in ranked
            if sc > 0
        ]


class DenseHybridRetriever:
    """Dense vector retrieval blended with BM25 (lexical). ``alpha`` weights the dense side."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        chunk_chars: int = 0,
        overlap: int = 0,
        alpha: float = 0.5,
    ):
        self.alpha = alpha
        self.dense = VectorRetriever(embedder, chunk_chars, overlap)
        self.bm25 = Bm25Retriever()
        self._articles: list[Article] = []

    def fit(self, articles: list[Article]) -> DenseHybridRetriever:
        self._articles = articles
        self.dense.fit(articles)
        self.bm25.fit(articles)
        return self

    def load(self, directory, articles: list[Article]) -> DenseHybridRetriever:
        """Load the persisted dense index (raises ValueError if stale); BM25 is rebuilt in-memory."""
        self.dense.load(directory, articles)
        self.bm25.fit(articles)
        self._articles = articles
        return self

    def query(self, question: str, top_k: int = 3) -> list[ScoredArticle]:
        n = len(self._articles)
        by_id: dict[int, float] = {}
        for weight, retr in ((self.alpha, self.dense), (1 - self.alpha, self.bm25)):
            for m in retr.query(question, top_k=n):
                key = m.article["article_number"]
                by_id[key] = by_id.get(key, 0.0) + weight * m.score
        lookup = {a["article_number"]: a for a in self._articles}
        ranked = sorted(by_id.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [ScoredArticle(lookup[k], s) for k, s in ranked if s > 0]


RETRIEVERS: dict[str, Callable[[], Retriever]] = {
    "tfidf": TfidfRetriever,
    "bm25": Bm25Retriever,
    "hybrid": HybridRetriever,
    "embedding": EmbeddingRetriever,
    "vector": VectorRetriever,
    "hybrid-dense": DenseHybridRetriever,
}


def build_retriever(name: str) -> Retriever:
    """Build by name. The dense retrievers read embedding_model / chunk_chars / chunk_overlap
    from settings (PRODML_EMBEDDING_MODEL, PRODML_CHUNK_CHARS, PRODML_CHUNK_OVERLAP)."""
    from prodml.config import settings

    if name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever {name!r}; choose from {sorted(RETRIEVERS)}")
    if name in ("vector", "hybrid-dense"):
        return RETRIEVERS[name](
            build_embedder(settings.embedding_model), settings.chunk_chars, settings.chunk_overlap
        )
    return RETRIEVERS[name]()
