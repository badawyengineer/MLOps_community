"""Validate a corpus before it ever reaches an embedding step.

A silent parsing bug here becomes a hallucination three steps later,
and by then it's the model that gets blamed. Run this on every corpus
rebuild (also enforced as a unit test in tests/test_corpus.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from prodml.corpus import Article

MAX_REASONABLE_ARTICLE_CHARS = 8000


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_corpus(articles: list[Article]) -> ValidationResult:
    result = ValidationResult()

    if not articles:
        result.errors.append("Corpus is empty.")
        return result

    numbers = [a["article_number"] for a in articles]
    if len(numbers) != len(set(numbers)):
        result.errors.append("Duplicate article_number values found.")

    sorted_numbers = sorted(numbers)
    gaps = [(a, b) for a, b in pairwise(sorted_numbers) if b - a > 1]
    if gaps:
        result.warnings.append(
            f"Non-contiguous article numbers — {len(gaps)} gap(s), "
            f"e.g. {gaps[:3]}. Confirm these are legitimately missing "
            "(not yet ingested) rather than a parsing skip."
        )

    for article in articles:
        n = article["article_number"]
        if not isinstance(n, int):
            result.errors.append(f"article_number is not an int: {n!r}")

        if not article["is_repealed"] and not article["ar_text"].strip():
            result.errors.append(f"Article {n}: empty ar_text and not flagged repealed.")

        if len(article["ar_text"]) > MAX_REASONABLE_ARTICLE_CHARS:
            result.warnings.append(
                f"Article {n}: ar_text is {len(article['ar_text'])} chars — "
                "suspiciously long, check for a failed article-boundary split."
            )

        if not article.get("citation"):
            result.errors.append(f"Article {n}: missing citation field.")

    return result


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import logging
    import sys

    from prodml.config import settings
    from prodml.corpus import load_corpus

    logging.basicConfig(level="INFO", format="%(message)s")
    log = logging.getLogger(__name__)

    articles = load_corpus(settings.corpus_path)
    result = validate_corpus(articles)
    for w in result.warnings:
        log.warning(w)
    for e in result.errors:
        log.error(e)
    if not result.ok:
        sys.exit(1)
    log.info("OK — %d articles validated.", len(articles))


if __name__ == "__main__":
    main()
