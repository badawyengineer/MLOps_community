"""Embedding backends behind one tiny interface: ``name``, ``fit(texts)``, ``encode(texts)``.

  * ``hashing``  - char n-gram feature hashing. Offline, stateless, deterministic: the default,
                   so the service and CI never need a model download.
  * ``lsa``      - TF-IDF + truncated SVD ("latent semantic analysis"). Offline, fitted.
  * any name containing ``/`` (e.g. ``intfloat/multilingual-e5-small``) -> sentence-transformers
                   (optional extra). Not exercised in the offline build environment.
All outputs are L2-normalized so cosine similarity is a dot product.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


class Embedder(Protocol):
    name: str
    stateless: bool

    def fit(self, texts: Sequence[str]) -> Embedder: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class HashingEmbedder:
    stateless = True

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"
        self._vec = HashingVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), n_features=dim, alternate_sign=False, norm=None
        )

    def fit(self, texts: Sequence[str]) -> HashingEmbedder:
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return _l2(self._vec.transform(list(texts)).toarray().astype(float))


class LsaEmbedder:
    stateless = False

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.name = f"lsa-{dim}"
        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: Sequence[str]) -> LsaEmbedder:
        matrix = self._tfidf.fit_transform(list(texts))
        k = max(1, min(self.dim, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=0).fit(matrix)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("LsaEmbedder.fit() must be called first.")
        return _l2(self._svd.transform(self._tfidf.transform(list(texts))))


def _prefixes_for(model_name: str) -> tuple[str, str]:
    """(document prefix, query prefix). The e5 family was trained with "passage: " / "query: "
    markers and loses accuracy without them; bge-m3, MiniLM, LaBSE and friends need none."""
    if "e5" in model_name.lower().split("/")[-1]:
        return "passage: ", "query: "
    return "", ""


class SentenceTransformerEmbedder:
    """Hugging Face sentence-transformers backend (optional extra, needs a model download).

    ``encode`` embeds DOCUMENTS (article text); ``encode_queries`` embeds user questions. They
    differ only for models that want asymmetric prefixes (see ``_prefixes_for``)."""

    stateless = True

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = None
        self.doc_prefix, self.query_prefix = _prefixes_for(model_name)

    def fit(self, texts: Sequence[str]) -> SentenceTransformerEmbedder:
        return self

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if self._model is None:  # pragma: no cover - needs the optional extra + a model download
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError('pip install -e ".[embeddings]"') from exc
            self._model = SentenceTransformer(self.name)
        batch = [prefix + t for t in texts]
        return _l2(np.asarray(self._model.encode(batch, normalize_embeddings=True), dtype=float))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.doc_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)


def build_embedder(spec: str) -> Embedder:
    """'hashing' | 'hashing-1024' | 'lsa' | 'lsa-32' | '<org>/<hf-model>'"""
    kind, _, dim = spec.partition("-")
    if kind == "hashing":
        return HashingEmbedder(int(dim) if dim else 512)
    if kind == "lsa":
        return LsaEmbedder(int(dim) if dim else 64)
    if "/" in spec:
        return SentenceTransformerEmbedder(spec)
    raise ValueError(f"Unknown embedding model {spec!r}")
