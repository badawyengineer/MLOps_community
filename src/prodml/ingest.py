"""Bilingual PDF -> per-article JSON (Step 0 of Track B).

Pipeline: PDF page -> words split into Arabic / English by SCRIPT (robust to column bleed) -> clean
(presentation forms, visual-order RTL, digits) -> article-boundary state machine that
also tracks Book/Chapter/Section headings -> merge the two languages by article number ->
fill declared repealed ranges with flagged placeholders.

Handles both layouts seen so far: shaped/visual-order Arabic (presentation forms) and plain
visual-order Arabic from Word exports (e.g. the real Civil Code PDF: reversed glyph order, lam-alef
ligature glyphs, bold headings, a promulgation-law preamble, and in-text "Articles 54-80 repealed"
notes). When the PDF carries font information, bold lines are headings and everything else is text.

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

from prodml._console import ensure_utf8_output
from prodml.corpus import Article, arabic_indic_to_western, normalize_arabic_text, write_corpus

logger = logging.getLogger(__name__)

_AR_CHAR = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_PRESENTATION_FORMS = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
_LTR_RUN = re.compile(r"[0-9A-Za-z٠-٩۰-۹]+")
_MIRROR = str.maketrans("()[]{}<>", ")(][}{><")

_BR = r"[\(\)\[\]]*"  # stray brackets: RTL extraction leaves them on either side of the number
_AR_ARTICLE = re.compile(rf"^\s*(?:ال)?مادة\s*{_BR}\s*([0-9٠-٩۰-۹]+)\s*{_BR}\s*[:.\-–]?\s*(.*)$")
_EN_ARTICLE = re.compile(rf"^\s*A?rticle\s*{_BR}\s*(\d+)\s*{_BR}\s*[:.\-–]?\s*(.*)$", re.IGNORECASE)
# The real code opens with the promulgation law (its own "Article 1/2"); the Code proper starts here.
_AR_START = re.compile(r"نصوص\s+القانون\s+المدن[يى]")
_ONLY_PUNCT = re.compile(r"^[\s()\[\].:\-–]*$")
# a marker line that is ONLY the marker ("Article 54", "مادة ) ٥)"): no period, no text after it
_BARE = re.compile(r"^[\s()\[\]]*$")
_BARE_LINE = re.compile(r"^(?:A?rticle|مادة)[0-9٠-٩۰-۹]+$", re.IGNORECASE)


def _is_bare_marker(line: str) -> bool:
    """'Article 54', 'Article 5 (', 'مادة ) ٥)' - but not 'Article 444.', 'المادة ٨١٦' (a
    cross-reference: wrapped sentences start with the definite article / end in a period)."""
    return bool(_BARE_LINE.match(re.sub(r"[\s()\[\]]+", "", line)))


# in-text notes such as "Articles 389-417 repealed" / "المواد من ٣٨٩ إلى ٤١٧ ملغاة"
_REPEAL_NOTE_EN = re.compile(
    r"^\W*Articles?\s+(\d+)\s*(?:[-–]|to)\s*(\d+)\s+(?:have\s+been\s+)?repealed\b", re.IGNORECASE
)
_REPEAL_NOTE_AR = re.compile(
    r"^\W*المواد\s+من\s+([0-9٠-٩۰-۹]+)\s+إل[ىي]\s+([0-9٠-٩۰-۹]+)(?:\s+ملغ\w*)?\W*$"
)
# numbered topic headings: "1. Laws and Rights", "١- القانون والحق"
_NUMBERED_TOPIC = re.compile(r"^\s*[0-9٠-٩۰-۹]+\s*[.\-–]")
_PUNCT_TOKEN = re.compile(r"^[().:;,\[\]\"'“”]+$")
_REPEALED = re.compile(r"ملغ[اىي]ة?|\bRepealed\b|\bAbrogated\b", re.IGNORECASE)

# heading label -> hierarchy level (EN "Chapter" = AR "باب", EN "Section" = AR "فصل" in this code)
_EN_HEADINGS = (
    (re.compile(r"^\s*(Book|Part)\b", re.IGNORECASE), "book"),
    (re.compile(r"^\s*(Title|Chapter)\b", re.IGNORECASE), "chapter"),
    (re.compile(r"^\s*(Section|Branch|Subsection)\b", re.IGNORECASE), "section"),
)
_AR_HEADINGS = (
    (re.compile(r"^\s*(?:ال)?(كتاب|قسم)\b"), "book"),
    (re.compile(r"^\s*(?:ال)?باب\b"), "chapter"),
    (re.compile(r"^\s*(?:ال)?(فصل|فرع|مبحث)\b"), "section"),
)


def looks_visual_order(text: str) -> bool:
    """Shaped Arabic (presentation forms) is what visual-order PDF text looks like."""
    return bool(_PRESENTATION_FORMS.search(text))


def unvisual(line: str) -> str:
    """Undo visual-order RTL extraction: reverse the line, restore the digit/Latin runs that
    were reversed along with it, and un-mirror brackets."""
    flipped = line[::-1].translate(_MIRROR)
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1], flipped)


# Word/Arabic-shaped PDFs expose each lam-alef ligature glyph as ONE pdfplumber char whose text is
# already the logical pair ("لا"). Reversing a visual-order word as a plain string would swap
# those two letters ("لا" -> "ال"), so each ligature char is swapped for its single-codepoint
# presentation form (still counted as Arabic, restored by NFKC after the line is un-reversed).
_LIGATURE_CHARS = {
    "\u0644\u0627": "\ufefb",  # لا
    "\u0644\u0623": "\ufef7",  # لأ
    "\u0644\u0625": "\ufef9",  # لإ
    "\u0644\u0622": "\ufef5",  # لآ
}
# frequent function words, forward vs. as they look when a visual-order line is not reversed
_FORWARD_WORDS = frozenset(["في", "من", "على", "أو", "التي", "الذي", "إلى", "عن", "هذا", "كل"])
_VISUAL_WORDS = frozenset(w[::-1] for w in _FORWARD_WORDS)


def detect_visual_order(lines: list[str]) -> bool | None:
    """True/False when extracted Arabic lines are clearly in visual/logical order, judged by
    whether common function words appear reversed more often than forward; None when there is no
    evidence (e.g. shaped presentation-form text, where ``looks_visual_order`` decides per line).
    Works for PDFs whose text layer has no presentation forms (Word exports)."""
    fwd = vis = 0
    for line in lines:
        for tok in line.split():
            fwd += tok in _FORWARD_WORDS
            vis += tok in _VISUAL_WORDS
    if fwd == vis:
        return None
    return vis > fwd


def clean_pdf_line(line: str, visual: bool | None = None) -> str:
    """Un-reverse visual-order text FIRST (so lam-alef ligatures are still single glyphs),
    then NFKC-fold presentation forms into logical letters, then normalize.

    ``visual`` forces (True) or forbids (False) the un-reversal; ``None`` falls back to the
    presentation-forms heuristic (``looks_visual_order``)."""
    if visual is None:
        visual = looks_visual_order(line) and len(_AR_CHAR.findall(line)) > len(line) * 0.3
    if visual and _AR_CHAR.search(line):
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
    topic: str = ""

    @property
    def text(self) -> str:
        return normalize_arabic_text(" ".join(self.lines))


_LEVELS = ("book", "chapter", "section", "topic")  # outermost -> innermost


def _tidy_label(label: str) -> str:
    """Heading labels come from bold lines like '1. Laws and Rights -': drop the dangling dash."""
    return re.sub(r"\s+", " ", label).strip(" -–:")


def parse_column(
    lines: list,
    lang: str,
    start: re.Pattern[str] | None = None,
    repealed_out: list[tuple[int, int]] | None = None,
) -> dict[int, ParsedArticle]:
    """Article-boundary state machine over one language's lines.

    ``lines`` is ``[(page, text), ...]`` or ``[(page, text, is_bold), ...]`` in reading order.
    Headings update the Book/Chapter/Section/topic context that every following article
    inherits. With the 3-tuple form (PDF font info available) headings are exactly the bold
    lines; with the 2-tuple form they are short lines that start with a heading keyword.

    ``start``: ignore everything before the first line matching it (promulgation-law preamble).
    ``repealed_out``: collects ranges announced by in-text notes ("Articles 389-417 repealed");
    the note and any footnote text after it are dropped instead of polluting the previous article.
    """
    article_re = _AR_ARTICLE if lang == "ar" else _EN_ARTICLE
    headings = _AR_HEADINGS if lang == "ar" else _EN_HEADINGS
    note_re = _REPEAL_NOTE_AR if lang == "ar" else _REPEAL_NOTE_EN
    items = [
        (it[0], it[1].strip(), bool(it[2]) if len(it) > 2 and it[2] is not None else None)
        for it in lines
    ]
    items = [it for it in items if it[1]]
    if not any(b for _, _, b in items):  # no bold anywhere: font info can't identify headings
        items = [(pg, t, None) for pg, t, _ in items]

    # If it never matches, parse the whole column rather than silently returning nothing.
    if start is not None and not any(start.search(t) for _, t, _ in items):
        logger.warning(
            "Start marker %r not found in %s column; parsing from the top", start.pattern, lang
        )
        start = None
    started = start is None

    # Real PDFs put the marker alone on its line; a line that merely STARTS with "Article 444."
    # is a wrapped cross-reference. If almost every marker in this column is bare, only bare
    # ones count; formats that put the text on the marker line ("Article 5: text") still work.
    shapes = [_is_bare_marker(t) for _, t, _ in items if article_re.match(t)]
    bare_only = bool(shapes) and sum(shapes) / len(shapes) >= 0.8

    out: dict[int, ParsedArticle] = {}
    ctx = dict.fromkeys(_LEVELS, "")
    current: ParsedArticle | None = None
    run_target: str | None = None  # where continuation lines of a bold run are appended
    skipping = False  # inside the footnote that follows a repeal note
    bold_body = (
        False  # the open article's body is set in bold (a typesetting glitch seen in the PDF)
    )

    def set_level(level: str, label: str) -> None:
        ctx[level] = label
        for deeper in _LEVELS[_LEVELS.index(level) + 1 :]:
            ctx[deeper] = ""

    for page, line, bold in items:
        if not started:
            started = bool(start and start.search(line))
            continue
        if bold is not True:
            run_target = None

        note = note_re.match(line)
        if note and (bold or bold is None or len(line) < 60):
            lo, hi = (int(arabic_indic_to_western(g)) for g in note.groups())
            if repealed_out is not None and lo <= hi:
                repealed_out.append((lo, hi))
            current, skipping = None, True
            continue

        m = article_re.match(line)
        if m and bare_only and not _is_bare_marker(line):
            # Inline marker ("Article 277 If the option ...") is accepted only as the very next
            # article and with text after it; "Article 444." / "المادة ٨١٦" are cross-references.
            nxt = int(arabic_indic_to_western(m.group(1)))
            inline = bool(m.group(2).strip()) and not line.lstrip().startswith("المادة")
            if not (inline and out and nxt == next(reversed(out)) + 1):
                m = None
        if m:
            skipping = False
            number = int(arabic_indic_to_western(m.group(1)))
            if number in out:
                logger.warning(
                    "Article %d appears twice (%s column); keeping the first", number, lang
                )
                current = None
                continue
            bold_body = False
            current = ParsedArticle(number, [], page, **{k: _tidy_label(v) for k, v in ctx.items()})
            out[number] = current
            if m.group(2).strip() and not _ONLY_PUNCT.match(m.group(2)):
                current.lines.append(m.group(2).strip())
            continue

        level = next((lvl for rx, lvl in headings if rx.match(line)), None)
        if bold is True and current is not None and not level and (not current.lines or bold_body):
            # bold text straight after a marker (before any body line) is the article itself
            current.lines.append(line)
            bold_body = True
            continue
        if bold is True:  # a bold line is a heading, a title continuation or a sub-topic
            skipping, current, bold_body = False, None, False
            if level:
                set_level(level, line)
                run_target = level
            elif run_target and run_target != "topic" and not _NUMBERED_TOPIC.match(line):
                ctx[run_target] = f"{ctx[run_target]} {line}"  # e.g. "SECTION I" + its title
            elif run_target == "topic" and not _NUMBERED_TOPIC.match(line):
                ctx["topic"] = f"{ctx['topic']} {line}"
            else:
                set_level("topic", line)
                run_target = "topic"
            continue
        if bold is None and level and len(line) < 120:  # no font info: keyword headings
            set_level(level, line)
            current = None  # heading text is not article text
            continue
        if skipping:
            continue
        if current is not None:
            bold_body = False
            current.lines.append(line)
    return out


def _is_repealed_text(text: str) -> bool:
    """A repealed article is a bare stub like '(Repealed)'. Long texts that merely mention
    repeal ('A provision of a law can only be repealed by a subsequent law ...') are live law."""
    return len(text) <= 80 and bool(_REPEALED.search(text))


def merge_columns(
    ar: dict[int, ParsedArticle], en: dict[int, ParsedArticle]
) -> tuple[list[Article], list[str]]:
    """Join the two languages on article number. Returns (articles, warnings)."""
    warnings: list[str] = []
    articles: list[Article] = []
    for n in sorted(set(ar) | set(en)):
        a, e = ar.get(n), en.get(n)
        text_ar, text_en = (a.text if a else ""), (e.text if e else "")
        repealed = _is_repealed_text(text_ar) or _is_repealed_text(text_en)
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
                "topic": ref.topic or section,
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


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of inclusive integer ranges: [(54, 80), (60, 90)] -> [(54, 90)]."""
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def drop_empty_in_ranges(articles: list[Article], ranges: list[tuple[int, int]]) -> list[Article]:
    """A repealed range's own marker ('Article 54' + a footnote) parses as an empty article;
    drop those so ``fill_repealed_ranges`` can replace them with one flagged placeholder."""

    def in_range(n: int) -> bool:
        return any(lo <= n <= hi for lo, hi in ranges)

    return [
        a
        for a in articles
        if not (in_range(a["article_number"]) and not a["text_ar"] and not a["text_en"])
    ]


def fill_repealed_ranges(articles: list[Article], ranges: list[tuple[int, int]]) -> list[Article]:
    """Insert flagged placeholders for declared repealed numbers that extraction did not
    produce, so the corpus never has a silent gap a RAG system could hallucinate around.
    A placeholder inherits book/chapter/section from the article just before it."""
    have = {a["article_number"] for a in articles}
    by_number = sorted(articles, key=lambda a: a["article_number"])
    extra: list[Article] = []
    for lo, hi in ranges:
        for n in range(lo, hi + 1):
            if n not in have:
                before = [a for a in by_number if a["article_number"] < n]
                ctx = before[-1] if before else None
                extra.append(
                    {
                        "article_number": n,
                        "book": ctx.get("book", "") if ctx else "",
                        "chapter": ctx.get("chapter", "") if ctx else "",
                        "section": ctx.get("section", "") if ctx else "",
                        "topic": "[Repealed]",
                        "text_ar": "",
                        "text_en": "[This article has been repealed.]",
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


def _classify_words(words: list[dict]) -> list[tuple[dict, str]]:
    """Words of one visual line sorted left-to-right, each tagged 'ar' or 'en' by SCRIPT, not by
    x position - so column bleed and left/right swaps between pages cannot mix the languages.
    Script-neutral tokens (digits, punctuation) join the nearest classified word."""
    words = sorted(words, key=lambda w: w["x0"])
    kinds = [_word_script(w["text"]) for w in words]
    for i, k in enumerate(kinds):
        if k == "amb":
            known = [j for j, kk in enumerate(kinds) if kk != "amb"]
            kinds[i] = kinds[min(known, key=lambda j: abs(j - i))] if known else "en"
    return list(zip(words, kinds, strict=True))


def split_line_by_script(words: list[dict]) -> tuple[str, str]:
    """Split one visual line's words into (arabic_text, english_text)."""
    tagged = _classify_words(words)
    ar = " ".join(w["text"] for w, k in tagged if k == "ar")
    # A stand-alone ".", "(" or ":" in the English column is the Arabic line's punctuation
    # leaking across the gutter ("was ." / "Article 1 ("): real English punctuation is attached.
    en = " ".join(w["text"] for w, k in tagged if k == "en" and not _PUNCT_TOKEN.match(w["text"]))
    return ar, en


def _is_bold(words: list[dict]) -> bool | None:
    """Mostly-bold line? None when the PDF gives no font names (then headings fall back to
    keyword matching)."""
    lettered = [w for w in words if re.search(r"\w", w["text"]) and "fontname" in w]
    if not lettered:
        return None
    return sum("bold" in w["fontname"].lower() for w in lettered) / len(lettered) >= 0.8


def _page_words(page, line_tolerance: float) -> list[list[dict]]:
    """Words of one page grouped into visual lines, with ligature glyphs kept as one char."""
    from pdfplumber.utils import extract_words

    chars = [{**c, "text": _LIGATURE_CHARS.get(c["text"], c["text"])} for c in page.chars]
    words = sorted(
        extract_words(chars, extra_attrs=["fontname"]), key=lambda w: (w["top"], w["x0"])
    )
    groups: list[list[dict]] = []
    for w in words:
        if groups and abs(w["top"] - groups[-1][0]["top"]) <= line_tolerance:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def extract_columns(
    pdf_path: Path,
    max_pages: int | None = None,
    line_tolerance: float = 3.0,
    visual: bool | None = None,
) -> tuple[list[tuple[int, str, bool | None]], list[tuple[int, str, bool | None]]]:
    """Return (arabic_lines, english_lines), each ``[(page, line, is_bold), ...]`` in reading order.

    ``visual``: whether the Arabic text layer is in visual order. ``None`` = auto-detect from
    the first pages (``detect_visual_order``), falling back to the per-line shaped-text check."""
    import pdfplumber

    raw: list[tuple[int, str, bool | None, str, bool | None]] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for page in pages:
            for group in _page_words(page, line_tolerance):
                tagged = _classify_words(group)
                ar, en = split_line_by_script(group)
                raw.append(
                    (
                        page.page_number,
                        ar,
                        _is_bold([w for w, k in tagged if k == "ar"]),
                        en,
                        _is_bold([w for w, k in tagged if k == "en"]),
                    )
                )
    if visual is None:
        visual = detect_visual_order([ar for _, ar, _, _, _ in raw[:400] if ar])
        logger.info(
            "Arabic text layer order: %s",
            {True: "VISUAL", False: "logical", None: "undetermined (per-line check)"}[visual],
        )
    ar_lines: list[tuple[int, str, bool | None]] = []
    en_lines: list[tuple[int, str, bool | None]] = []
    for page_no, ar, ar_bold, en, en_bold in raw:
        if ar:
            ar_lines.append((page_no, clean_pdf_line(ar, visual=visual), ar_bold))
        if en:
            en_lines.append((page_no, clean_pdf_line(en, visual=False), en_bold))
    return ar_lines, en_lines


def apply_manual_fixes(
    articles: list[Article], fixes: list[dict]
) -> tuple[list[Article], list[str]]:
    """Apply documented corrections for defects in the SOURCE PDF (``data/manual_fixes.json``).

    A fix replaces ``field`` of one article with ``new`` only when the extracted value still equals
    ``expect_current``; otherwise the extractor or the PDF changed since the fix was written and it
    is reported as stale and NOT applied. Returns (articles, messages)."""
    by_number = {a["article_number"]: a for a in articles}
    messages: list[str] = []
    for fix in fixes:
        n, field_name = fix["article_number"], fix["field"]
        art = by_number.get(n)
        if art is None or art.get(field_name) != fix["expect_current"]:
            messages.append(
                f"manual fix for article {n} {field_name} is STALE (extracted text differs): not applied"
            )
            continue
        art[field_name] = fix["new"]  # type: ignore[literal-required]
        messages.append(f"manual fix applied: article {n} {field_name}")
    return articles, messages


def load_manual_fixes(path: Path) -> list[dict]:
    import json

    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("fixes", []))


def build_articles_from_pdf(
    pdf_path: Path,
    repealed_ranges: list[tuple[int, int]] | None = None,
    max_pages: int | None = None,
) -> list[Article]:
    """PDF -> per-article records. Repealed ranges come from ``repealed_ranges`` (config) UNIONED
    with the ranges the PDF itself announces ("Articles 389-417 repealed")."""
    ar_lines, en_lines = extract_columns(pdf_path, max_pages)
    announced: list[tuple[int, int]] = []
    ar = parse_column(ar_lines, "ar", _AR_START, repealed_out=announced)
    en = parse_column(en_lines, "en", repealed_out=announced)
    articles, warnings = merge_columns(ar, en)
    for w in warnings[:20]:
        logger.warning(w)
    if len(warnings) > 20:
        logger.warning("... and %d more column-mismatch warnings", len(warnings) - 20)
    ranges = merge_ranges((repealed_ranges or []) + announced)
    if announced:
        logger.info("Repeal notes found in the PDF: %s", merge_ranges(announced))
    articles = fill_repealed_ranges(drop_empty_in_ranges(articles, ranges), ranges)
    # data/raw/<pdf> -> data/manual_fixes.json (documented corrections for defects in the source PDF)
    articles, fix_msgs = apply_manual_fixes(
        articles, load_manual_fixes(pdf_path.parent.parent / "manual_fixes.json")
    )
    for m in fix_msgs:
        logger.warning(m) if "STALE" in m else logger.info(m)
    return articles


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_output()
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
        print(
            "=== ARABIC COLUMN ===",
            *[f"[p{pg}{'*' if b else ''}] {ln}" for pg, ln, b in ar],
            sep="\n",
        )
        print(
            "=== ENGLISH COLUMN ===",
            *[f"[p{pg}{'*' if b else ''}] {ln}" for pg, ln, b in en],
            sep="\n",
        )
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
