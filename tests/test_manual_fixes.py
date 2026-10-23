"""Documented corrections for source-PDF defects are applied only when they still fit."""

from __future__ import annotations

from prodml.ingest import apply_manual_fixes


def _art(n: int, ar: str) -> dict:
    return {"article_number": n, "text_ar": ar, "text_en": "x", "is_repealed": False}


def test_fix_applies_when_expectation_matches() -> None:
    arts, msgs = apply_manual_fixes(
        [_art(1022, "")],
        [{"article_number": 1022, "field": "text_ar", "expect_current": "", "new": "نص"}],
    )
    assert arts[0]["text_ar"] == "نص" and "applied" in msgs[0]


def test_stale_fix_is_reported_and_not_forced() -> None:
    arts, msgs = apply_manual_fixes(
        [_art(1022, "already fixed upstream")],
        [{"article_number": 1022, "field": "text_ar", "expect_current": "", "new": "نص"}],
    )
    assert arts[0]["text_ar"] == "already fixed upstream" and "STALE" in msgs[0]


def test_fix_for_missing_article_is_stale() -> None:
    _, msgs = apply_manual_fixes(
        [], [{"article_number": 5, "field": "text_ar", "expect_current": "", "new": "x"}]
    )
    assert "STALE" in msgs[0]
