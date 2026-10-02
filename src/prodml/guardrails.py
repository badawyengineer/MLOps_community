"""Output guardrails, applied to every generated answer before it leaves the service.

* PII redaction - emails, phone numbers (Egyptian + international), Egyptian national IDs
  (14 digits), card-like numbers. Statute text has no PII, so anything matching is either
  echoed from a user question or invented by the model; neither should be returned.
* Citation check - the model may only cite articles that were actually retrieved. A
  hallucinated "[Article 999]" is the single most damaging failure for a legal assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from prodml.corpus import arabic_indic_to_western

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("national_id", re.compile(r"(?<!\d)[23]\d{13}(?!\d)")),
    ("card", re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?20|0)?1[0125]\d{8}(?!\d)|(?<!\d)\+\d{1,3}[ -]?\d{6,12}(?!\d)"),
    ),
]
_CITATION = re.compile(r"\[?\bArticle\s+(\d+)\]?|\[?(?:ال)?مادة\s*(\d+)\]?", re.IGNORECASE)


@dataclass
class GuardResult:
    text: str
    flags: list[str] = field(default_factory=list)
    unsupported_citations: list[int] = field(default_factory=list)


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, kinds_found). Arabic-Indic digits are matched too."""
    western = arabic_indic_to_western(text)
    found: list[str] = []
    out = western
    for kind, pattern in _PII_PATTERNS:
        out, n = pattern.subn(f"[REDACTED-{kind.upper()}]", out)
        if n:
            found.append(kind)
    # keep the original text untouched if nothing matched (don't silently rewrite digits)
    return (out, found) if found else (text, [])


def cited_articles(text: str) -> set[int]:
    western = arabic_indic_to_western(text)
    return {int(a or b) for a, b in _CITATION.findall(western)}


def apply_guardrails(answer: str, allowed_articles: set[int]) -> GuardResult:
    redacted, kinds = redact_pii(answer)
    result = GuardResult(redacted, [f"pii:{k}" for k in kinds])
    bad = sorted(cited_articles(redacted) - allowed_articles)
    if bad:
        result.unsupported_citations = bad
        result.flags.append("unsupported_citation")
    return result
