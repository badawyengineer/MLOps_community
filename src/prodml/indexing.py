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


def build_index(allow_sample: bool = False) -> VectorRetriever:
    """Build and persist the index. A missing corpus is an ERROR unless ``allow_sample`` is set:
    silently indexing the 5-article sample during an image build would ship the wrong data."""
    if settings.corpus_path.exists():
        articles = load_corpus(settings.corpus_path)
    elif allow_sample:
        articles = sample_corpus()
    else:
        raise FileNotFoundError(
            f"No corpus at {settings.corpus_path}. Run `prodml-ingest` (real PDF) or "
            "`python -m prodml.corpus` (sample), or pass --allow-sample."
        )
    retriever = VectorRetriever(
        build_embedder(settings.embedding_model), settings.chunk_chars, settings.chunk_overlap
    )
    retriever.fit(articles)
    retriever.save(settings.index_dir)
    return retriever


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO", format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--allow-sample", action="store_true", help="index the 5-article sample if no corpus exists"
    )
    args = ap.parse_args(argv)
    r = build_index(args.allow_sample)
    assert r.store is not None
    logging.getLogger(__name__).info("Indexed %s -> %s", r.store.meta, settings.index_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
