"""Bilingual PDF -> per-article JSON (Step 0 of Track B).

Pipeline: PDF page -> words split into Arabic / English by SCRIPT (robust to column bleed) -> clean
(presentation forms, visual-order RTL, digits) -> article-boundary state machine that
also tracks Book/Chapter/Section headings -> merge the two languages by article number ->
fill declared repealed ranges with flagged placeholders.

IMPORTANT: layouts differ between PDF editions. The pure-text functions below are
unit-tested; the column geometry is tested only against a *synthetic* fixture. Before
trusting a real run, dump the columns and read them:

    python -m prodml.ingest data/raw/egyptian_civil_code.pdf --dump 3

and fix whatever looks wrong (heading patterns in ``_EN_HEADINGS`` / ``_AR_HEADINGS``, the
article-marker regexes, ``line_tolerance``) before running the full conversion.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from prodml.corpus import Article, arabic_indic_to_western, normalize_arabic_text, write_corpus

logger = logging.getLogger(__name__)

_AR_CHAR = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_PRESENTATION_FORMS = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
_LTR_RUN = re.compile(r"[0-9A-Za-z٠-٩۰-۹]+")
_MIRROR = str.maketrans("()[]{}<>", ")(][}{><")

_AR_ARTICLE = re.compile(r"^\s*(?:ال)?مادة\s*[\(\[]?\s*([0-9٠-٩۰-۹]+)\s*[\)\]]?\s*[:.\-–]?\s*(.*)$")
_EN_ARTICLE = re.compile(
    r"^\s*Article\s*[\(\[]?\s*(\d+)\s*[\)\]]?\s*[:.\-–]?\s*(.*)$", re.IGNORECASE
)
_REPEALED = re.compile(r"ملغ[اىي]ة?|\bRepealed\b|\bAbrogated\b", re.IGNORECASE)

# heading label -> hierarchy level
_EN_HEADINGS = (
    (re.compile(r"^\s*(Book|Part)\b", re.IGNORECASE), "book"),
    (re.compile(r"^\s*(Title|Chapter)\b", re.IGNORECASE), "chapter"),
    (re.compile(r"^\s*(Section|Branch|Subsection)\b", re.IGNORECASE), "section"),
)
_AR_HEADINGS = (
    (re.compile(r"^\s*(?:ال)?(كتاب|قسم)\b"), "book"),
    (re.compile(r"^\s*(?:ال)?(باب|فصل)\b"), "chapter"),
    (re.compile(r"^\s*(?:ال)?(فرع|مبحث)\b"), "section"),
)


def looks_visual_order(text: str) -> bool:
    """Shaped Arabic (presentation forms) is what visual-order PDF text looks like."""
    return bool(_PRESENTATION_FORMS.search(text))


def unvisual(line: str) -> str:
    """Undo visual-order RTL extraction: reverse the line, restore the digit/Latin runs that
    were reversed along with it, and un-mirror brackets."""
    flipped = line[::-1].translate(_MIRROR)
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1], flipped)


def clean_pdf_line(line: str) -> str:
    """Un-reverse visual-order text FIRST (so lam-alef ligatures are still single glyphs),
    then NFKC-fold presentation forms into logical letters, then normalize."""
    if looks_visual_order(line) and len(_AR_CHAR.findall(line)) > len(line) * 0.3:
        line = unvisual(line)
    return normalize_arabic_text(unicodedata.normalize("NFKC", line))


@dataclass
class ParsedArticle:
    number: int
    lines: list[str] = field(default_factory=list)
    page: int | None = None
    book: str = ""
    chapter: str = ""
    section: str = ""

    @property
    def text(self) -> str:
        return normalize_arabic_text(" ".join(self.lines))


def parse_column(lines: list[tuple[int, str]], lang: str) -> dict[int, ParsedArticle]:
    """Article-boundary state machine over one language's lines.

    ``lines`` is ``[(page_number, text), ...]`` in reading order. Headings update the
    Book/Chapter/Section context that every following article inherits.
    """
    article_re = _AR_ARTICLE if lang == "ar" else _EN_ARTICLE
    headings = _AR_HEADINGS if lang == "ar" else _EN_HEADINGS
    out: dict[int, ParsedArticle] = {}
    ctx = {"book": "", "chapter": "", "section": ""}
    current: ParsedArticle | None = None

    for page, raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = article_re.match(line)
        if m:
            number = int(arabic_indic_to_western(m.group(1)))
            if number in out:
                logger.warning(
                    "Article %d appears twice (%s column); keeping the first", number, lang
                )
                current = None
                continue
            current = ParsedArticle(number, [], page, **ctx)
            out[number] = current
            if m.group(2).strip():
                current.lines.append(m.group(2).strip())
            continue
        heading = next((lvl for rx, lvl in headings if rx.match(line)), None)
        if heading and len(line) < 120:
            ctx[heading] = line
            if heading == "book":
                ctx["chapter"] = ctx["section"] = ""
            elif heading == "chapter":
                ctx["section"] = ""
            current = None  # heading text is not article text
            continue
        if current is not None:
            current.lines.append(line)
    return out


def merge_columns(
    ar: dict[int, ParsedArticle], en: dict[int, ParsedArticle]
) -> tuple[list[Article], list[str]]:
    """Join the two languages on article number. Returns (articles, warnings)."""
    warnings: list[str] = []
    articles: list[Article] = []
    for n in sorted(set(ar) | set(en)):
        a, e = ar.get(n), en.get(n)
        text_ar, text_en = (a.text if a else ""), (e.text if e else "")
        repealed = bool(_REPEALED.search(text_ar) or _REPEALED.search(text_en))
        if not a:
            warnings.append(f"Article {n}: found in English column only")
        if not e:
            warnings.append(f"Article {n}: found in Arabic column only")
        ref = e or a
        assert ref is not None
        section = ref.section or ref.chapter or ref.book
        articles.append(
            {
                "article_number": n,
                "book": ref.book,
                "chapter": ref.chapter,
                "section": ref.section,
                "topic": section,
                "text_ar": text_ar,
                "text_en": text_en,
                "is_repealed": repealed,
                "source_page": ref.page,
                "citation": f"Egyptian Civil Code, Article {n}"
                + (" (repealed)" if repealed else ""),
            }
        )
    return articles, warnings


def parse_ranges(spec: str) -> list[tuple[int, int]]:
    """'54-80,100' -> [(54, 80), (100, 100)]"""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        out.append((int(lo), int(hi or lo)))
    return out


def fill_repealed_ranges(articles: list[Article], ranges: list[tuple[int, int]]) -> list[Article]:
    """Insert flagged placeholders for declared repealed numbers that extraction did not
    produce, so the corpus never has a silent gap a RAG system could hallucinate around."""
    have = {a["article_number"] for a in articles}
    extra: list[Article] = []
    for lo, hi in ranges:
        for n in range(lo, hi + 1):
            if n not in have:
                extra.append(
                    {
                        "article_number": n,
                        "book": "",
                        "chapter": "",
                        "section": "",
                        "topic": "[Repealed]",
                        "text_ar": "",
                        "text_en": "[This article was repealed by Presidential Decree.]",
                        "is_repealed": True,
                        "source_page": None,
                        "citation": f"Egyptian Civil Code, Article {n} (repealed)",
                    }
                )
    return sorted(articles + extra, key=lambda a: a["article_number"])


def _word_script(word: str) -> str:
    if _AR_CHAR.search(word) or re.search("[٠-٩۰-۹]", word):
        return "ar"
    return "en" if re.search("[A-Za-z]", word) else "amb"


def split_line_by_script(words: list[dict]) -> tuple[str, str]:
    """Split one visual line's words into (arabic_text, english_text) by SCRIPT, not by x
    position - so column bleed and left/right swaps between pages cannot mix the languages.
    Script-neutral tokens (digits, punctuation) join the nearest classified word."""
    words = sorted(words, key=lambda w: w["x0"])
    kinds = [_word_script(w["text"]) for w in words]
    for i, k in enumerate(kinds):
        if k == "amb":
            known = [j for j, kk in enumerate(kinds) if kk != "amb"]
            kinds[i] = kinds[min(known, key=lambda j: abs(j - i))] if known else "en"
    ar = " ".join(w["text"] for w, k in zip(words, kinds, strict=True) if k == "ar")
    en = " ".join(w["text"] for w, k in zip(words, kinds, strict=True) if k == "en")
    return ar, en


def extract_columns(
    pdf_path: Path, max_pages: int | None = None, line_tolerance: float = 3.0
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (arabic_lines, english_lines), each ``[(page, line), ...]`` in reading order."""
    import pdfplumber

    ar_lines: list[tuple[int, str]] = []
    en_lines: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for page in pages:
            words = sorted(page.extract_words(), key=lambda w: (w["top"], w["x0"]))
            groups: list[list[dict]] = []
            for w in words:
                if groups and abs(w["top"] - groups[-1][0]["top"]) <= line_tolerance:
                    groups[-1].append(w)
                else:
                    groups.append([w])
            for group in groups:
                ar, en = split_line_by_script(group)
                if ar:
                    ar_lines.append((page.page_number, clean_pdf_line(ar)))
                if en:
                    en_lines.append((page.page_number, clean_pdf_line(en)))
    return ar_lines, en_lines


def build_articles_from_pdf(
    pdf_path: Path,
    repealed_ranges: list[tuple[int, int]] | None = None,
    max_pages: int | None = None,
) -> list[Article]:
    ar_lines, en_lines = extract_columns(pdf_path, max_pages)
    articles, warnings = merge_columns(parse_column(ar_lines, "ar"), parse_column(en_lines, "en"))
    for w in warnings[:20]:
        logger.warning(w)
    if len(warnings) > 20:
        logger.warning("... and %d more column-mismatch warnings", len(warnings) - 20)
    return fill_repealed_ranges(articles, repealed_ranges or [])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO", format="%(message)s")
    from prodml.config import settings

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("pdf", type=Path, nargs="?", default=settings.raw_pdf_path)
    p.add_argument(
        "--dump", type=int, metavar="N", help="print both columns of the first N pages and exit"
    )
    p.add_argument("--repealed", default=settings.repealed_ranges, help="e.g. '54-80'")
    p.add_argument("--out", type=Path, default=settings.corpus_path)
    args = p.parse_args(argv)

    if not args.pdf.exists():
        logger.error("No PDF at %s", args.pdf)
        return 1
    if args.dump:
        ar, en = extract_columns(args.pdf, args.dump)
        print("=== ARABIC COLUMN ===", *[f"[p{pg}] {ln}" for pg, ln in ar], sep="\n")
        print("=== ENGLISH COLUMN ===", *[f"[p{pg}] {ln}" for pg, ln in en], sep="\n")
        return 0
    articles = build_articles_from_pdf(args.pdf, parse_ranges(args.repealed))
    write_corpus(articles, args.out)
    logger.info(
        "Wrote %d articles to %s - now run: python -m prodml.validate_corpus --strict",
        len(articles),
        args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
