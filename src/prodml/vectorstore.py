"""A small persistent vector store: exact cosine search over L2-normalized vectors.

For ~1,150 articles exact search is instant, needs no server, and the whole index is two
files (``vectors.npz`` + ``meta.json``) that can be baked into the Docker image and tracked
by DVC. No pickle anywhere: arrays go through npz, metadata through JSON. Swap in
Qdrant/Chroma behind this same interface if the corpus ever outgrows it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    def __init__(
        self, vectors: np.ndarray, payloads: list[dict[str, Any]], meta: dict[str, Any]
    ) -> None:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")
        self.vectors, self.payloads, self.meta = vectors, payloads, meta

    def __len__(self) -> int:
        return len(self.payloads)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[dict[str, Any], float]]:
        if not len(self):
            return []
        scores = self.vectors @ query
        top = np.argsort(scores)[::-1][:k]
        return [(self.payloads[i], float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "vectors.npz", vectors=self.vectors)
        (directory / "meta.json").write_text(
            json.dumps({"meta": self.meta, "payloads": self.payloads}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        vectors = np.load(directory / "vectors.npz", allow_pickle=False)["vectors"]
        blob = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(vectors, blob["payloads"], blob["meta"])
