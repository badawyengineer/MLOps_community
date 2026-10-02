"""Build and persist the vector index (``prodml-index``): corpus JSON -> chunk by article ->
embed -> vector store on disk. Baked into the Docker image and a DVC stage."""

from __future__ import annotations

import argparse
import logging
import sys

from prodml.config import settings
from prodml.corpus import load_corpus, sample_corpus
from prodml.embedders import build_embedder
from prodml.retriever import VectorRetriever


def build_index() -> VectorRetriever:
    articles = (
        load_corpus(settings.corpus_path) if settings.corpus_path.exists() else sample_corpus()
    )
    retriever = VectorRetriever(
        build_embedder(settings.embedding_model), settings.chunk_chars, settings.chunk_overlap
    )
    retriever.fit(articles)
    retriever.save(settings.index_dir)
    return retriever


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO", format="%(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    r = build_index()
    assert r.store is not None
    logging.getLogger(__name__).info("Indexed %s -> %s", r.store.meta, settings.index_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
