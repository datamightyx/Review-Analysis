"""Convert a DOCX export of an Amazon review page into a text-only PDF.

Amazon review pages are normally saved as browser-print PDFs (BUCATSTATE.pdf,
Petchardom.pdf, ...) — that is the only shape `pipeline/pdf_parser.py` reads.
A DOCX save of the same page carries the same review text plus review photos
and UI leftovers ("Click to play video", "Translate review to English"). This
script drops the images and that UI noise and re-emits the reviews as a plain
PDF laid out exactly like the browser prints:

  <Author>
  <R>.0 out of 5 stars <Title>
  Reviewed in the United States on <Month D, YYYY>
  Size: ...Color: ...Verified Purchase
  <body>
  [N people found this helpful]
  Helpful
  Report

Usage:  python scripts/docx_reviews_to_pdf.py <in.docx> <out.pdf>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from fontTools.ttLib import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont
from reportlab.pdfgen import canvas

# Amazon page furniture that carries no review content. The parser already
# strips most of these, but "Translate all reviews to English" is not in its
# NOISE_PATTERNS and would be swallowed into the NEXT review's author.
NOISE_RE = re.compile(
    r"^(?:Click to play video"
    r"|Translate (?:all )?reviews? to English"
    r"|Video Player is loading.*"
    r"|Images? in this review"
    r"|See more reviews?"
    r"|Read more"
    r"|Accessibility support for this content.*)$",
    re.I,
)

FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_NAME = "ReviewSans"
FONT_SIZE = 10
LEADING = 12.5
MARGIN = 54          # 0.75in
PAGE_W, PAGE_H = letter


def _font_charset(path: Path) -> set[int]:
    """Codepoints the TTF can actually render. Characters outside it (the
    Thai/box-drawing bits of kaomoji display names) are dropped rather than
    emitted as .notdef, which would extract back as garbage."""
    with TTFont(str(path), fontNumber=0) as tt:
        return set(tt.getBestCmap())


def _normalise(text: str, charset: set[int]) -> str:
    text = text.replace("\xa0", " ").replace("\u2060", "")
    text = "".join(c for c in text if c == "\n" or ord(c) in charset)
    return text.rstrip()


def docx_lines(docx_path: Path) -> list[str]:
    """Review lines from the DOCX, noise and blanks removed. Embedded images
    are simply never visited — python-docx paragraphs hold text only."""
    charset = _font_charset(FONT_PATH)
    lines: list[str] = []
    for para in Document(str(docx_path)).paragraphs:
        for raw in _normalise(para.text, charset).split("\n"):
            line = raw.strip()
            if not line or NOISE_RE.match(line):
                continue
            lines.append(line)
    return lines


def _wrap(canv: canvas.Canvas, line: str, width: float) -> list[str]:
    out, cur = [], ""
    for word in line.split(" "):
        trial = f"{cur} {word}".strip()
        if cur and canv.stringWidth(trial, FONT_NAME, FONT_SIZE) > width:
            out.append(cur)
            cur = word
        else:
            cur = trial
    out.append(cur)
    return out


def write_pdf(lines: list[str], pdf_path: Path) -> None:
    pdfmetrics.registerFont(RLTTFont(FONT_NAME, str(FONT_PATH)))
    canv = canvas.Canvas(str(pdf_path), pagesize=letter)
    canv.setFont(FONT_NAME, FONT_SIZE)
    width = PAGE_W - 2 * MARGIN
    y = PAGE_H - MARGIN
    for line in lines:
        for piece in _wrap(canv, line, width):
            if y < MARGIN:
                canv.showPage()
                canv.setFont(FONT_NAME, FONT_SIZE)
                y = PAGE_H - MARGIN
            canv.drawString(MARGIN, y, piece)
            y -= LEADING
    canv.save()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    lines = docx_lines(src)
    write_pdf(lines, dst)
    print(f"{src.name} -> {dst.name}: {len(lines)} lines, "
          f"{sum(1 for l in lines if l.startswith('Reviewed in'))} reviews")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
