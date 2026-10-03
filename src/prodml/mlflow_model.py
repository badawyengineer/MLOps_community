"""MLflow pyfunc wrapper so the winning retrieval configuration can live in the Model Registry
and be loaded back (``models:/legal-qa-retriever@champion``) exactly as it was evaluated."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from prodml.embedders import build_embedder
from prodml.retriever import VectorRetriever


class RetrievalConfigModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        cfg = json.loads(Path(context.artifacts["config"]).read_text(encoding="utf-8"))
        articles = json.loads(Path(context.artifacts["corpus"]).read_text(encoding="utf-8"))
        self.retriever = VectorRetriever(
            build_embedder(cfg["embedding_model"]), cfg["chunk_size"], cfg["overlap"]
        ).fit(articles)

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        questions = list(model_input["question"])
        rows = []
        for q in questions:
            top = self.retriever.query(q, top_k=1)
            rows.append(
                {
                    "citation": top[0].article["citation"] if top else "",
                    "score": top[0].score if top else 0.0,
                }
            )
        return pd.DataFrame(rows)
