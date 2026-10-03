"""Generates tests/fixtures/bilingual_sample.pdf - a SYNTHETIC two-column Arabic/English PDF.

It reproduces the failure modes the handbook warns about (shaped presentation-form glyphs in
visual order, Arabic-Indic article numbers, a repealed article, heading hierarchy) so the
extractor's mechanics are testable. It is NOT the real Civil Code layout.
Requires: pip install -e ".[fixtures]"
"""

from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
pdfmetrics.registerFont(TTFont("DejaVu", FONT))

AR_PAGE1 = [
    "الكتاب الأول",
    "الالتزامات",
    "مادة ١٤٧",
    "العقد شريعة المتعاقدين فلا يجوز نقضه",
    "ولا تعديله إلا باتفاق الطرفين.",
    "مادة ١٤٨",
    "يجب تنفيذ العقد طبقا لما اشتمل عليه",
    "وبطريقة تتفق مع ما يوجبه حسن النية.",
]
EN_PAGE1 = [
    "Book Two: Obligations",
    "Chapter One: Contracts",
    "Article 147",
    "The contract makes the law of the parties.",
    "It can be revoked or altered only by mutual consent.",
    "Article 148",
    "A contract must be performed in accordance with its",
    "contents and in good faith.",
]
AR_PAGE2 = [
    "الفصل الثاني",
    "مادة ١٦٣",
    "كل خطأ سبب ضررا للغير يلزم من ارتكبه بالتعويض.",
    "مادة ١٦٤ (ملغاة)",
]
EN_PAGE2 = [
    "Chapter Two: Tortious Liability",
    "Article 163",
    "Any fault which causes damage to another obliges",
    "the person who committed it to make reparation.",
    "Article 164 (Repealed)",
]


def shaped(line: str) -> str:
    return get_display(arabic_reshaper.reshape(line))


def draw(c, lines, x, rtl):
    c.setFont("DejaVu", 11)
    y = 780
    for ln in lines:
        txt = shaped(ln) if rtl else ln
        c.drawRightString(x, y, txt) if rtl else c.drawString(x, y, txt)
        y -= 22


def main(out: Path) -> None:
    c = canvas.Canvas(str(out), pagesize=(595, 842))
    # page 1: English LEFT, Arabic RIGHT
    draw(c, EN_PAGE1, 30, False)
    draw(c, AR_PAGE1, 565, True)
    c.showPage()
    # page 2: sides swapped, to prove per-page script detection
    draw(c, AR_PAGE2, 280, True)
    draw(c, EN_PAGE2, 310, False)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main(Path(__file__).parent / "bilingual_sample.pdf")
