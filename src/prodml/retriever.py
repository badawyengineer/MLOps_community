"""Retrieval layer.

Module 1 ships a dependency-light TF-IDF retriever so the whole service
is runnable and testable offline, with no external model download.
Module 2's MLflow experiments (chunk_size / overlap / embedding_model)
are exactly where this gets swapped for a real multilingual embedding
model (e.g. a sentence-transformers/AraBERT encoder) and benchmarked
against this baseline — keep this class's interface stable so that
swap doesn't ripple into the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from prodml.corpus import Article, normalize_for_search


@dataclass
class ScoredArticle:
    article: Article
    score: float


class TfidfRetriever:
    """Fits one TF-IDF index over normalized article text and returns
    the top-k most similar articles to a query, by cosine similarity.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._matrix = None
        self._articles: list[Article] = []

    def fit(self, articles: list[Article]) -> TfidfRetriever:
        self._articles = articles
        corpus_text = [
            normalize_for_search(f"{a['topic']} {a['ar_text']} {a['text_en']}") for a in articles
        ]
        self._matrix = self._vectorizer.fit_transform(corpus_text)
        return self

    def query(self, question: str, top_k: int = 3) -> list[ScoredArticle]:
        if self._matrix is None:
            raise RuntimeError("TfidfRetriever.fit() must be called before query().")

        query_vec = self._vectorizer.transform([normalize_for_search(question)])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        ranked_idx = scores.argsort()[::-1][:top_k]
        return [
            ScoredArticle(article=self._articles[i], score=float(scores[i]))
            for i in ranked_idx
            if scores[i] > 0
        ]
