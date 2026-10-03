from __future__ import annotations

from pathlib import Path

import pytest

from prodml.corpus import build_from_pdf
from prodml.ingest import (
    build_articles_from_pdf,
    clean_pdf_line,
    fill_repealed_ranges,
    merge_columns,
    parse_column,
    parse_ranges,
    split_line_by_script,
    unvisual,
)
from prodml.validate_corpus import spot_check, validate_corpus

FIXTURE = Path(__file__).parent / "fixtures" / "bilingual_sample.pdf"


def test_parse_ranges() -> None:
    assert parse_ranges("54-80,100") == [(54, 80), (100, 100)]
    assert parse_ranges("") == []


def test_unvisual_flips_word_order_but_keeps_runs_and_brackets_intact() -> None:
    # visual "(abc) 147" == logical "147 (abc)": order flips, runs and bracket orientation survive
    assert unvisual("(abc) 147") == "147 (abc)"
    assert unvisual(unvisual("(abc) 147")) == "(abc) 147"  # involution


def test_clean_pdf_line_handles_shaped_visual_arabic() -> None:
    import arabic_reshaper
    from bidi.algorithm import get_display

    shaped = get_display(arabic_reshaper.reshape("مادة ١٤٧ لا يجوز"))
    assert clean_pdf_line(shaped) == "مادة ١٤٧ لا يجوز"


def test_clean_pdf_line_leaves_plain_text_alone() -> None:
    assert clean_pdf_line("Article  147 ") == "Article 147"


def test_parse_column_tracks_headings_and_multiline_text() -> None:
    lines = [
        (1, "Book Two: Obligations"),
        (1, "Chapter One: Contracts"),
        (1, "Article 147"),
        (1, "The contract makes the law"),
        (2, "of the parties."),
        (2, "Article (148): Second text on same line"),
    ]
    out = parse_column(lines, "en")
    assert out[147].text == "The contract makes the law of the parties."
    assert out[147].page == 1 and out[147].chapter == "Chapter One: Contracts"
    assert out[148].text == "Second text on same line"


def test_parse_column_arabic_indic_numbers_and_headings() -> None:
    out = parse_column([(1, "الكتاب الأول"), (1, "مادة ١٤٧"), (1, "نص المادة")], "ar")
    assert list(out) == [147] and out[147].book == "الكتاب الأول"


def test_new_book_resets_chapter_and_section() -> None:
    out = parse_column(
        [
            (1, "Chapter One"),
            (1, "Section A"),
            (1, "Article 1"),
            (1, "x"),
            (2, "Book Two"),
            (2, "Article 2"),
            (2, "y"),
        ],
        "en",
    )
    assert out[1].chapter == "Chapter One" and out[2].chapter == "" and out[2].section == ""


def test_duplicate_article_keeps_first() -> None:
    out = parse_column([(1, "Article 5"), (1, "first"), (1, "Article 5"), (1, "second")], "en")
    assert out[5].text == "first"


def test_merge_flags_repealed_and_one_sided_articles() -> None:
    ar = parse_column([(1, "مادة ٣"), (1, "(ملغاة)")], "ar")
    en = parse_column(
        [(1, "Article 3"), (1, "(Repealed)"), (1, "Article 4"), (1, "only english")], "en"
    )
    articles, warnings = merge_columns(ar, en)
    by_n = {a["article_number"]: a for a in articles}
    assert by_n[3]["is_repealed"] and by_n[3]["citation"].endswith("(repealed)")
    assert any("Article 4" in w and "English column only" in w for w in warnings)


def test_fill_repealed_ranges_adds_flagged_placeholders() -> None:
    base = [
        {"article_number": 53, "is_repealed": False},
        {"article_number": 55, "is_repealed": False},
    ]
    out = fill_repealed_ranges(base, [(54, 56)])  # type: ignore[arg-type]
    assert [a["article_number"] for a in out] == [53, 54, 55, 56]
    assert out[1]["is_repealed"] and out[3]["is_repealed"] and not out[2]["is_repealed"]


def test_split_line_by_script_assigns_neutral_tokens_to_neighbours() -> None:
    words = [
        {"text": "Article", "x0": 10},
        {"text": "147", "x0": 60},
        {"text": "مادة", "x0": 400},
        {"text": "(", "x0": 450},
    ]
    ar, en = split_line_by_script(words)
    assert en == "Article 147" and ar == "مادة" + " (" or ar.startswith("مادة")


def test_pdf_end_to_end_on_synthetic_fixture() -> None:
    articles = build_articles_from_pdf(FIXTURE, parse_ranges("150-151"))
    by_n = {a["article_number"]: a for a in articles}
    assert sorted(by_n) == [147, 148, 150, 151, 163, 164]
    assert by_n[147]["text_ar"].startswith("العقد شريعة المتعاقدين فلا يجوز نقضه")
    assert by_n[147]["text_en"].startswith("The contract makes the law of the parties.")
    assert by_n[147]["book"].startswith("Book Two") and by_n[163]["chapter"].startswith(
        "Chapter Two"
    )
    assert by_n[163]["source_page"] == 2 and by_n[147]["source_page"] == 1
    assert by_n[164]["is_repealed"] and by_n[150]["is_repealed"] and not by_n[148]["is_repealed"]


def test_corpus_entrypoint_delegates_to_ingest() -> None:
    assert any(a["article_number"] == 163 for a in build_from_pdf(FIXTURE))


def test_strict_validation_rejects_unexplained_gaps_but_lenient_warns() -> None:
    articles = build_articles_from_pdf(FIXTURE)  # 147,148,163,164 -> gaps
    assert validate_corpus(articles).ok
    strict = validate_corpus(articles, strict=True)
    assert not strict.ok and any("unexplained gap" in e for e in strict.errors)


def test_strict_validation_passes_when_gaps_are_declared_repealed() -> None:
    articles = build_articles_from_pdf(FIXTURE, parse_ranges("149-162"))
    assert validate_corpus(articles, strict=True).ok


def test_spot_check_renders_requested_count() -> None:
    articles = build_articles_from_pdf(FIXTURE)
    text = spot_check(articles, n=3)
    assert text.count("--- Article") == 3


@pytest.mark.parametrize("bad", ["", "zzz"])
def test_extract_requires_nothing_for_missing_headings(bad: str) -> None:
    assert parse_column([(1, bad)], "en") == {}
