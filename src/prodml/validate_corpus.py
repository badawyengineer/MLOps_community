"""Validate a corpus before it ever reaches an embedding step.

A silent parsing bug here becomes a hallucination three steps later,
and by then it's the model that gets blamed. Run this on every corpus
rebuild (also enforced as a unit test in tests/test_corpus.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from prodml._console import ensure_utf8_output
from prodml.corpus import Article

MAX_REASONABLE_ARTICLE_CHARS = 8000


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_corpus(
    articles: list[Article],
    strict: bool = False,
    known_issues: dict[int, str] | None = None,
) -> ValidationResult:
    """``strict`` (use for the full corpus): unexplained gaps in article numbering are errors,
    and every non-repealed article must carry both Arabic and English text.

    ``known_issues`` maps an article number to a documented defect in the SOURCE document (e.g.
    the Arabic text of an article is simply absent from the PDF). Such an article's missing text
    is reported as a warning carrying the reason instead of failing the build - the defect stays
    visible, but it is not mistaken for a parsing bug."""
    known_issues = known_issues or {}
    result = ValidationResult()

    if not articles:
        result.errors.append("Corpus is empty.")
        return result

    numbers = [a["article_number"] for a in articles]
    if len(numbers) != len(set(numbers)):
        result.errors.append("Duplicate article_number values found.")

    sorted_numbers = sorted(numbers)
    gaps = [(a, b) for a, b in pairwise(sorted_numbers) if b - a > 1]
    if gaps and strict:
        result.errors.append(
            f"{len(gaps)} unexplained gap(s) in article numbering, e.g. {gaps[:3]}. Gaps must be "
            "either ingested or declared repealed (PRODML_REPEALED_RANGES) - never silent."
        )
    elif gaps:
        result.warnings.append(
            f"Non-contiguous article numbers — {len(gaps)} gap(s), "
            f"e.g. {gaps[:3]}. Confirm these are legitimately missing "
            "(not yet ingested) rather than a parsing skip."
        )

    for article in articles:
        n = article["article_number"]
        if not isinstance(n, int):
            result.errors.append(f"article_number is not an int: {n!r}")

        if not article["is_repealed"] and not article["text_ar"].strip():
            if n in known_issues:
                result.warnings.append(
                    f"Article {n}: empty text_ar - known source defect: {known_issues[n]}"
                )
            else:
                result.errors.append(f"Article {n}: empty text_ar and not flagged repealed.")

        if len(article["text_ar"]) > MAX_REASONABLE_ARTICLE_CHARS:
            result.warnings.append(
                f"Article {n}: text_ar is {len(article['text_ar'])} chars — "
                "suspiciously long, check for a failed article-boundary split."
            )

        if strict and not article["is_repealed"] and not article["text_en"].strip():
            if n in known_issues:
                result.warnings.append(
                    f"Article {n}: empty text_en - known source defect: {known_issues[n]}"
                )
            else:
                result.errors.append(
                    f"Article {n}: empty text_en (extraction lost the English column)."
                )

        if len(article["text_en"]) > MAX_REASONABLE_ARTICLE_CHARS:
            result.warnings.append(
                f"Article {n}: text_en is {len(article['text_en'])} chars - check the split."
            )

        if not article.get("citation"):
            result.errors.append(f"Article {n}: missing citation field.")

    return result


def load_known_issues(path: Path | None = None) -> dict[int, str]:
    """``data/processed/known_issues.json``: {"1022": "reason", ...} - defects in the source PDF."""
    import json

    from prodml.config import settings

    path = path or settings.corpus_path.with_name("known_issues.json")
    if not path.exists():
        return {}
    return {int(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def spot_check(articles: list[Article], n: int = 20, seed: int = 0) -> str:
    """Render n random articles side by side for the by-eye check the handbook requires
    ("eyeball 20 random articles before you trust the whole file")."""
    import random

    sample = random.Random(seed).sample(articles, min(n, len(articles)))
    blocks = []
    for a in sorted(sample, key=lambda x: x["article_number"]):
        blocks.append(
            f"--- Article {a['article_number']} (p{a['source_page']}) repealed={a['is_repealed']}\n"
            f"{a['book']} > {a['chapter']} > {a['section']}\nAR: {a['text_ar'][:300]}\nEN: {a['text_en'][:300]}"
        )
    return "\n\n".join(blocks)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    ensure_utf8_output()
    import argparse
    import logging
    import sys

    from prodml.config import settings
    from prodml.corpus import load_corpus

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict", action="store_true", help="full-corpus rules (no unexplained gaps)"
    )
    parser.add_argument(
        "--spot-check", type=int, metavar="N", help="print N random articles to eyeball"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(message)s")
    log = logging.getLogger(__name__)

    articles = load_corpus(settings.corpus_path)
    if args.spot_check:
        print(spot_check(articles, args.spot_check))
    result = validate_corpus(articles, strict=args.strict, known_issues=load_known_issues())
    for w in result.warnings:
        log.warning(w)
    for e in result.errors:
        log.error(e)
    if not result.ok:
        sys.exit(1)
    log.info("OK - %d articles validated.", len(articles))


if __name__ == "__main__":
    main()
