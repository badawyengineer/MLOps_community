"""Layout quirks of the real bilingual Civil Code PDF, reproduced on synthetic line lists."""

from __future__ import annotations

from prodml.ingest import (
    _AR_START,
    clean_pdf_line,
    detect_visual_order,
    drop_empty_in_ranges,
    fill_repealed_ranges,
    merge_columns,
    merge_ranges,
    parse_column,
)


def test_visual_order_is_detected_from_reversed_function_words() -> None:
    assert detect_visual_order(["يف نم ىلع", "نوناقلا يف"]) is True
    assert detect_visual_order(["في من على", "القانون في"]) is False
    assert detect_visual_order(["no arabic here"]) is None


def test_lam_alef_ligature_survives_unreversing() -> None:
    # visual order of "لا يجوز" with the ligature glyph as one presentation-form char
    assert clean_pdf_line("زوجي \ufefb", visual=True) == "لا يجوز"


def test_bold_lines_are_headings_and_titles_attach_to_them() -> None:
    lines = [
        (1, "BOOK II", True),
        (1, "SPECIFIC CONTRACTS", True),
        (1, "1. Sale in General -", True),
        (1, "Article 418", False),
        (1, "Sale is a contract.", False),
    ]
    a = parse_column(lines, "en")[418]
    assert a.book == "BOOK II SPECIFIC CONTRACTS"
    assert a.topic == "1. Sale in General"
    assert a.text == "Sale is a contract."


def test_cross_references_are_not_markers_but_inline_next_article_is() -> None:
    # a real column is overwhelmingly bare markers, which is what switches strict mode on
    lines: list[tuple[int, str, bool | None]] = []
    for n in range(270, 276):
        lines += [(1, f"Article {n}", False), (1, f"text of {n}", False)]
    lines += [
        (1, "Article 276", False),
        (1, "Article 277 If the option belongs to the debtor", False),  # inline, next in sequence
        (1, "see Article 444.", False),  # cross-reference mid-text
        (1, "Article 444.", False),  # wrapped cross-reference, not bare
        (1, "Article 278", False),
        (1, "text of 278", False),
    ]
    out = parse_column(lines, "en")
    assert list(out) == [*range(270, 279)]
    assert out[277].text.startswith("If the option")


def test_arabic_cross_reference_with_definite_article_is_ignored() -> None:
    lines: list[tuple[int, str, bool | None]] = []
    for n in "١٢٣٤٥":
        lines += [(1, f"مادة {n}", False), (1, "نص", False)]
    lines.insert(3, (1, "المادة ٨١٦", False))
    assert list(parse_column(lines, "ar")) == [1, 2, 3, 4, 5]


def test_repeal_note_collects_range_and_drops_footnote() -> None:
    found: list[tuple[int, int]] = []
    lines = [
        (1, "Article 388", False),
        (1, "last live text", False),
        (1, "Articles 389-417 repealed", True),
        (1, "footnote that must not become article text", False),
        (1, "Article 418", False),
        (1, "Sale", False),
    ]
    out = parse_column(lines, "en", repealed_out=found)
    assert found == [(389, 417)]
    assert out[388].text == "last live text"
    assert out[418].text == "Sale"


def test_bold_body_right_after_marker_is_article_text() -> None:
    lines = [(1, "مادة ٥٥٨", False), (1, "الإيجار عقد يلتزم المؤجر", True), (1, "مادة ٥٥٩", False)]
    assert "الإيجار" in parse_column(lines, "ar")[558].text


def test_preamble_before_start_marker_is_skipped() -> None:
    lines = [
        (1, "مادة ١", False),
        (1, "قانون الإصدار", False),
        (1, "نصوص القانون المدنى", False),
        (2, "مادة ١", False),
        (2, "النص الحقيقي", False),
    ]
    assert parse_column(lines, "ar", _AR_START)[1].text == "النص الحقيقي"


def test_repealed_range_stub_is_replaced_by_inheriting_placeholders() -> None:
    en = parse_column(
        [
            (1, "BOOK I", True),
            (1, "Article 53", False),
            (1, "live", False),
            (1, "Article 54", False),
        ],
        "en",
    )
    arts, _ = merge_columns({}, en)
    arts = fill_repealed_ranges(drop_empty_in_ranges(arts, [(54, 56)]), [(54, 56)])
    assert [a["article_number"] for a in arts] == [53, 54, 55, 56]
    assert all(a["is_repealed"] and a["book"] == "BOOK I" for a in arts[1:])


def test_long_text_mentioning_repeal_is_not_flagged_repealed() -> None:
    en = parse_column(
        [
            (1, "Article 2", False),
            (1, "A provision of a law can only be repealed by a later law " * 3, False),
        ],
        "en",
    )
    assert merge_columns({}, en)[0][0]["is_repealed"] is False


def test_merge_ranges_unions_overlaps() -> None:
    assert merge_ranges([(54, 80), (60, 90), (389, 417)]) == [(54, 90), (389, 417)]
