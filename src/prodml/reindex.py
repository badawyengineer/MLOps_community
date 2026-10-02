"""Batch re-indexing: add or update documents WITHOUT re-ingesting the PDF.

    python -m prodml.reindex new_articles.json            # merge + rewrite corpus + rebuild index
    python -m prodml.reindex new_articles.json --dry-run  # show what would change

``new_articles.json`` is a JSON list of article records in the corpus schema. Articles with an
existing ``article_number`` REPLACE the old record (an amendment); new numbers are added. The
merged corpus is re-validated before anything is written, and the vector index is rebuilt (it is
fingerprinted to the corpus, so a stale index is refused at load time anyway).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from prodml.config import settings
from prodml.corpus import Article, load_corpus, sample_corpus, write_corpus
from prodml.validate_corpus import validate_corpus

logger = logging.getLogger(__name__)


def merge_articles(
    existing: list[Article], incoming: list[Article]
) -> tuple[list[Article], dict[str, list[int]]]:
    by_n = {a["article_number"]: a for a in existing}
    added = [a["article_number"] for a in incoming if a["article_number"] not in by_n]
    replaced = [a["article_number"] for a in incoming if a["article_number"] in by_n]
    by_n.update({a["article_number"]: a for a in incoming})
    return [by_n[n] for n in sorted(by_n)], {"added": added, "replaced": replaced}


def reindex(new_path: Path, dry_run: bool = False) -> dict[str, list[int]]:
    existing = (
        load_corpus(settings.corpus_path) if settings.corpus_path.exists() else sample_corpus()
    )
    incoming = json.loads(new_path.read_text(encoding="utf-8"))
    merged, changes = merge_articles(existing, incoming)
    result = validate_corpus(merged)
    if not result.ok:
        raise ValueError("merged corpus failed validation: " + "; ".join(result.errors))
    if dry_run:
        return changes
    write_corpus(merged, settings.corpus_path)
    from prodml.indexing import build_index

    build_index()
    return changes


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO", format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("new_articles", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        changes = reindex(args.new_articles, args.dry_run)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        logger.error("reindex failed, nothing written: %s", exc)
        return 1
    logger.info(
        "%s: added=%s replaced=%s",
        "dry-run" if args.dry_run else "reindexed",
        changes["added"],
        changes["replaced"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
