from __future__ import annotations

import copy

import pytest

from prodml.corpus import (
    arabic_indic_to_western,
    normalize_arabic_text,
    normalize_for_search,
    sample_corpus,
    strip_diacritics,
)
from prodml.validate_corpus import validate_corpus


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("١٤٧", "147"),
        ("Article ٦٠", "Article 60"),
        ("no digits here", "no digits here"),
        ("", ""),
    ],
)
def test_arabic_indic_to_western(raw: str, expected: str) -> None:
    assert arabic_indic_to_western(raw) == expected


def test_strip_diacritics_removes_harakat() -> None:
    diacritized = "الْعَقْدُ شَرِيعَةُ الْمُتَعَاقِدِينَ"
    cleaned = strip_diacritics(diacritized)
    assert "\u064e" not in cleaned  # fatha
    assert "\u0652" not in cleaned  # sukun
    assert "العقد" in cleaned.replace(" ", "") or "لعقد" in cleaned


def test_normalize_arabic_text_collapses_whitespace() -> None:
    text = "العقد   شريعة\n\nالمتعاقدين"
    assert normalize_arabic_text(text) == "العقد شريعة المتعاقدين"


def test_normalize_for_search_folds_alef_and_ta_marbuta() -> None:
    assert normalize_for_search("أحمد") == normalize_for_search("احمد")
    assert normalize_for_search("مدرسة").endswith("سه")


def test_sample_corpus_is_well_formed() -> None:
    articles = sample_corpus()
    assert len(articles) >= 5
    numbers = [a["article_number"] for a in articles]
    assert len(numbers) == len(set(numbers)), "article numbers must be unique"


def test_validate_corpus_passes_on_sample_corpus() -> None:
    result = validate_corpus(sample_corpus())
    assert result.ok, result.errors


def test_validate_corpus_flags_empty_text_when_not_repealed() -> None:
    articles = copy.deepcopy(sample_corpus())
    articles[0]["ar_text"] = ""
    articles[0]["is_repealed"] = False

    result = validate_corpus(articles)

    assert not result.ok
    assert any("empty ar_text" in e for e in result.errors)


def test_validate_corpus_allows_empty_text_when_repealed() -> None:
    articles = copy.deepcopy(sample_corpus())
    articles[0]["ar_text"] = ""
    articles[0]["is_repealed"] = True

    result = validate_corpus(articles)

    assert result.ok, result.errors


def test_validate_corpus_flags_duplicate_article_numbers() -> None:
    articles = copy.deepcopy(sample_corpus())
    articles[1]["article_number"] = articles[0]["article_number"]

    result = validate_corpus(articles)

    assert not result.ok
    assert any("Duplicate" in e for e in result.errors)


def test_validate_corpus_rejects_empty_list() -> None:
    result = validate_corpus([])
    assert not result.ok
