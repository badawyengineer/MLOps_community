"""e5 models need asymmetric prefixes; other models must be left alone. No model download needed."""

from __future__ import annotations

import numpy as np

from prodml.embedders import SentenceTransformerEmbedder, _prefixes_for


class _FakeModel:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts, normalize_embeddings=True):
        self.seen = list(texts)
        return np.ones((len(self.seen), 4))


def test_prefixes_by_model_family() -> None:
    assert _prefixes_for("intfloat/multilingual-e5-small") == ("passage: ", "query: ")
    assert _prefixes_for("intfloat/multilingual-e5-large") == ("passage: ", "query: ")
    assert _prefixes_for("BAAI/bge-m3") == ("", "")
    assert _prefixes_for("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") == ("", "")


def test_documents_and_queries_get_different_prefixes() -> None:
    emb = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    emb._model = _FakeModel()
    emb.encode(["article text"])
    assert emb._model.seen == ["passage: article text"]
    emb.encode_queries(["a question"])
    assert emb._model.seen == ["query: a question"]


def test_models_without_prefixes_embed_text_unchanged() -> None:
    emb = SentenceTransformerEmbedder("BAAI/bge-m3")
    emb._model = _FakeModel()
    emb.encode_queries(["a question"])
    assert emb._model.seen == ["a question"]
