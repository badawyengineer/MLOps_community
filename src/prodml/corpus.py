"""Corpus ingestion — PDF -> structured per-article JSON.

This is Step 0 for the Track B final project: the corpus is raw input,
not the corpus. Everything downstream (chunking, embedding, retrieval,
citation) depends on getting this right.

Two entry points:

  * ``build_from_pdf(pdf_path)``   — real ingestion, requires the actual
    Egyptian Civil Code PDF and the optional ``pdf`` extra
    (``pip install -e ".[pdf]"``). Column-aware / RTL-aware extraction is
    genuinely hard and PDF-layout dependent, so this function is a
    documented starting point, not a finished parser — see the docstring
    on ``build_from_pdf`` for what to expect to hand-fix per source PDF.

  * ``sample_corpus()``            — a small, hand-verified set of real
    articles used to develop and test the rest of the pipeline (Modules
    1-5) *before* the full ~1,150-article corpus is ingested. Every
    article in it is genuine statutory text (Law No. 131 of 1948), not
    synthetic filler — statutes are not creative works, so quoting them
    verbatim carries none of the copyright concerns a literary or news
    source would.

Run ``python -m prodml.corpus`` (or the ``prodml-build-corpus`` console
script) to (re)write ``data/processed/articles.json`` from whichever
source is available.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import TypedDict

from prodml.config import settings

logger = logging.getLogger(__name__)

# Arabic-Indic digits -> Western digits, so article_number can be an int
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Extended (Persian/Urdu) digits, which show up in PDF extractions from mixed-font sources
_EXTENDED_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Arabic diacritics (harakat, tanween, sukun, shadda, quranic marks)
_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


class Article(TypedDict):
    article_number: int
    book: str
    chapter: str
    section: str
    topic: str
    text_ar: str
    text_en: str
    is_repealed: bool
    source_page: int | None
    citation: str


def arabic_indic_to_western(text: str) -> str:
    """Normalize Arabic-Indic numerals (١٤٧) to Western digits (147)."""
    return text.translate(_ARABIC_INDIC_DIGITS)


def strip_diacritics(text: str) -> str:
    """Remove harakat/tanween/shadda so the same word doesn't embed as
    two different tokens depending on whether a source happened to
    diacritize it."""
    return _DIACRITICS_RE.sub("", text)


def normalize_arabic_text(text: str) -> str:
    """Canonical cleanup applied to every extracted Arabic field:
    NFC-normalize the unicode, strip diacritics, collapse whitespace.
    Deliberately does NOT fold alef/hamza/ta-marbuta variants here —
    that fold is a *search-index* transform (see retriever.py), not a
    transform we want baked into the citable ``text_ar`` field.
    """
    text = unicodedata.normalize("NFC", text)
    text = strip_diacritics(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_search(text: str) -> str:
    """Aggressive normalization used only for retrieval indexing/queries:
    folds alef variants (أ إ آ -> ا) and ta-marbuta (ة -> ه) on top of
    ``normalize_arabic_text``, so a query typed without a hamza still
    matches an article that has one.
    """
    text = normalize_arabic_text(text)
    text = text.replace("\u0640", "")  # tatweel (kashida) elongation
    text = arabic_indic_to_western(text)
    text = text.translate(_EXTENDED_DIGITS)  # Persian/Urdu digit forms
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ک", "ك").replace("ی", "ي")  # Persian kaf / yeh -> Arabic
    return text.lower()


def sample_corpus() -> list[Article]:
    """A small, hand-verified bootstrap corpus of real articles from the
    Egyptian Civil Code (Law No. 131 of 1948), used to build and test
    Modules 1-5 before the full PDF is ingested.

    Replace this with ``build_from_pdf(settings.raw_pdf_path)`` once the
    real source document is available locally (see the README's
    "Growing the corpus" section).
    """
    return [
        {
            "article_number": 1,
            "book": "Preliminary Title",
            "chapter": "General Provisions",
            "section": "Application of the Law",
            "topic": "Sources of law in the absence of legislative text",
            "text_ar": normalize_arabic_text(
                "في حالة عدم وجود نص تشريعي يمكن تطبيقه، يحكم القاضي بمقتضى "
                "العرف، فإذا لم يوجد فبمقتضى مبادئ الشريعة الإسلامية، فإذا "
                "لم توجد فبمقتضى مبادئ القانون الطبيعي وقواعد العدالة."
            ),
            "text_en": (
                "In the absence of any applicable legislative text, the judge "
                "shall decide according to custom, and failing custom, "
                "according to the principles of Islamic law, and failing "
                "those, according to the principles of natural law and the "
                "rules of equity."
            ),
            "is_repealed": False,
            "source_page": 1,
            "citation": "Egyptian Civil Code, Article 1",
        },
        {
            "article_number": 147,
            "book": "Obligations or Personal Rights",
            "chapter": "Sources of Obligations",
            "section": "Contracts",
            "topic": "The Effects of a Contract — pacta sunt servanda",
            "text_ar": normalize_arabic_text(
                "العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق "
                "الطرفين أو للأسباب التي يقررها القانون."
            ),
            "text_en": (
                "The contract makes the law of the parties. It can be "
                "revoked or altered only by mutual consent of both parties, "
                "or for reasons provided for by law."
            ),
            "is_repealed": False,
            "source_page": 34,
            "citation": "Egyptian Civil Code, Article 147",
        },
        {
            "article_number": 148,
            "book": "Obligations or Personal Rights",
            "chapter": "Sources of Obligations",
            "section": "Contracts",
            "topic": "Performance of contracts in good faith",
            "text_ar": normalize_arabic_text(
                "يجب تنفيذ العقد طبقا لما اشتمل عليه وبطريقة تتفق مع ما " "يوجبه حسن النية."
            ),
            "text_en": (
                "A contract must be performed in accordance with its "
                "contents and in a manner consistent with the requirements "
                "of good faith."
            ),
            "is_repealed": False,
            "source_page": 34,
            "citation": "Egyptian Civil Code, Article 148",
        },
        {
            "article_number": 163,
            "book": "Obligations or Personal Rights",
            "chapter": "Sources of Obligations",
            "section": "Tortious Liability",
            "topic": "General rule of civil liability for wrongful acts",
            "text_ar": normalize_arabic_text("كل خطأ سبب ضررا للغير يلزم من ارتكبه بالتعويض."),
            "text_en": (
                "Any fault which causes damage to another obliges the "
                "person who committed it to make reparation."
            ),
            "is_repealed": False,
            "source_page": 39,
            "citation": "Egyptian Civil Code, Article 163",
        },
        {
            "article_number": 60,
            "book": "Persons",
            "chapter": "Juristic Persons",
            "section": "(repealed range)",
            "topic": "[Repealed — placeholder]",
            "text_ar": "",
            "text_en": (
                "[This article was repealed by a subsequent Presidential "
                "Decree; it is retained here, flagged, so the corpus does "
                "not silently have a gap that a RAG system might "
                "hallucinate around.]"
            ),
            "is_repealed": True,
            "source_page": None,
            "citation": "Egyptian Civil Code, Article 60 (repealed)",
        },
    ]


def build_from_pdf(pdf_path: Path) -> list[Article]:
    """PDF -> Article extraction for the bilingual Egyptian Civil Code PDF
    (implementation lives in ``prodml.ingest``).

    This is intentionally a *starting point*, not a finished parser —
    the handbook is explicit that this step eats an evening. Known
    failure modes to expect and hand-fix once you point this at the
    real file:

      * Two-column bilingual layout bleeds together under naive text
        extraction. You will likely need ``pdfplumber``'s
        ``page.extract_words()`` with x-coordinates to split columns,
        rather than ``page.extract_text()``.
      * RTL glyph runs can come out reversed or disconnected depending
        on the PDF's internal text order — spot-check ~20 random
        articles by eye after extraction, don't trust it blind.
      * Article numbers may be Arabic-Indic digits — always route them
        through ``arabic_indic_to_western`` before ``int()``.
      * Article boundaries: match on a regex for the heading pattern
        the source PDF actually uses (e.g. ``r"^مادة\\s+(\\d+)"`` or
        ``r"^Article\\s+(\\d+)"``) rather than assuming one article per
        page.

    Raises FileNotFoundError with a clear message if the PDF isn't
    present yet, rather than failing deep inside a parsing loop.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"No source PDF at {pdf_path}. Either place the Egyptian Civil "
            "Code PDF there, or use sample_corpus() to keep developing "
            "against the bootstrap dataset."
        )

    try:
        from prodml.ingest import build_articles_from_pdf, parse_ranges
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("PDF ingestion needs the 'pdf' extra: pip install -e '.[pdf]'") from exc

    return build_articles_from_pdf(pdf_path, repealed_ranges=parse_ranges(settings.repealed_ranges))


def write_corpus(articles: list[Article], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus(path: Path) -> list[Article]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:  # console-script entry point: prodml-build-corpus
    logging.basicConfig(level="INFO", format="%(message)s")
    if settings.raw_pdf_path.exists():
        articles = build_from_pdf(settings.raw_pdf_path)
    else:
        logger.warning(
            "No PDF found at %s; writing the sample bootstrap corpus instead.",
            settings.raw_pdf_path,
        )
        articles = sample_corpus()
    write_corpus(articles, settings.corpus_path)
    logger.info("Wrote %d articles to %s", len(articles), settings.corpus_path)


if __name__ == "__main__":
    main()
