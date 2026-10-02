"""Query-embedding drift: are users asking about the same kinds of things the system was
built and evaluated on?

Baseline = centroid of the embedded evaluation questions. Each live query's cosine to that
centroid is observed; a falling rolling mean means the question distribution is moving away
from what was tested (new topics, a different dialect, an upstream client change...).
Uses a dedicated stateless embedder so it is independent of whichever retriever is active.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from prodml.corpus import normalize_for_search
from prodml.embedders import HashingEmbedder


class EmbeddingDrift:
    def __init__(self, baseline_questions: list[str], window: int = 100) -> None:
        self._embedder = HashingEmbedder(512)
        centroid = self._embed(baseline_questions).mean(axis=0)
        self._centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        self._recent: deque[float] = deque(maxlen=window)
        self.baseline_mean = float(np.mean(self._embed(baseline_questions) @ self._centroid))

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._embedder.encode([normalize_for_search(t) for t in texts])

    def observe(self, question: str) -> float:
        cos = float(self._embed([question])[0] @ self._centroid)
        self._recent.append(cos)
        return cos

    @property
    def rolling_mean(self) -> float:
        return float(np.mean(self._recent)) if self._recent else self.baseline_mean

    @property
    def drop_from_baseline(self) -> float:
        return self.baseline_mean - self.rolling_mean
