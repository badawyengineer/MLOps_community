"""Chunk BY ARTICLE, not by token count.

A legal article is a self-contained unit with a natural citation, so the default
(``max_chars=0``) is one chunk per article. Long articles may be split on sentence
boundaries into overlapping chunks; EVERY chunk keeps the article number and citation, so
retrieval can always cite the article. ``max_chars`` / ``overlap`` are the knobs the MLflow
chunking experiments sweep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prodml.corpus import Article, normalize_for_search

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟۔;؛])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    article_number: int
    citation: str
    text: str
    index: int  # position within its article


def article_text(a: Article) -> str:
    return f"{a['topic']}. {a['text_ar']} {a['text_en']}".strip()


def _pack(units: list[str], max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) + 1 > max_chars:
            chunks.append(" ".join(current))
            # carry trailing units totalling <= overlap chars into the next chunk
            carry: list[str] = []
            carried = 0
            for u in reversed(current):
                if carried + len(u) + 1 > overlap:
                    break
                carry.insert(0, u)
                carried += len(u) + 1
            current, size = carry, carried
        current.append(unit)
        size += len(unit) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_article(a: Article, max_chars: int = 0, overlap: int = 0) -> list[Chunk]:
    if max_chars < 0 or overlap < 0 or (max_chars and overlap >= max_chars):
        raise ValueError("need 0 <= overlap < max_chars (max_chars=0 means whole-article chunks)")
    text = article_text(a)
    if max_chars == 0 or len(text) <= max_chars:
        pieces = [text]
    else:
        units = [u.strip() for u in _SENTENCE_SPLIT.split(text) if u.strip()]
        pieces = _pack(units, max_chars, overlap)
    n = a["article_number"]
    return [Chunk(f"{n}#{i}", n, a["citation"], p, i) for i, p in enumerate(pieces)]


def chunk_corpus(articles: list[Article], max_chars: int = 0, overlap: int = 0) -> list[Chunk]:
    return [c for a in articles for c in chunk_article(a, max_chars, overlap)]


def chunk_index_texts(chunks: list[Chunk]) -> list[str]:
    """What actually gets embedded: the search-normalized chunk text."""
    return [normalize_for_search(c.text) for c in chunks]
