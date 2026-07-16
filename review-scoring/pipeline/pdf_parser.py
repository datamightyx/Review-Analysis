"""Parse Amazon review-page PDF exports into structured Review objects.

The PDFs are browser prints of Amazon review pages. After whitespace
normalisation each review looks like:

  <Author> <R>.0 out of 5 stars <Title> Reviewed in the United States on
  <Month D, YYYY> [Size/Color/...: <variant>] [Verified Purchase]
  [Click to play video] <Body> [N people found this helpful] Helpful Report
"""
from __future__ import annotations

import re
from datetime import datetime, date
from pathlib import Path

from .models import Review

# "stars?": real exports occasionally print "5.0 out of 5 star" (no "s") —
# the recognised/expected counter caught a review lost to exactly that
RATING_RE = re.compile(r"(\d(?:[.,]\d)?)\s*out\s*of\s*5\s*stars?", re.I)
DATE_RE = re.compile(
    r"Reviewed in (?:the )?[\w\s]+? on\s+([A-Z][a-z]+ \d{1,2}, \d{4})"
)
VARIANT_RE = re.compile(
    r"\b(?:Size|Color|Colour|Style|Pattern|Flavor|Flavour|Scent|Number of Items|Package Quantity)\s*(?:Name)?\s*:\s*"
)
TRAILER_RE = re.compile(
    r"(?:\d+\s+people found this helpful\s*|One person found this helpful\s*)?"
    r"Helpful\s*\|?\s*Report(?:\s+abuse)?",
    re.I,
)
NOISE_PATTERNS = [
    r"Click to play video",
    r"Video Player is loading.*?(?:Transcript|$)",
    r"\d+\s+people found this helpful",
    r"One person found this helpful",
    r"Helpful\s*\|?\s*Report(?:\s+abuse)?",
    r"Read more",
    r"See more reviews?",
    r"Translate review to English",
]


def _extract_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        if text.strip():
            return text
    except Exception:
        pass
    import PyPDF2
    reader = PyPDF2.PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    for pat in NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _parse_date(s: str) -> str:
    try:
        return datetime.strptime(s, "%B %d, %Y").date().isoformat()
    except ValueError:
        return ""


def parse_pdf(pdf_path: Path, product: str,
              stats: dict | None = None) -> list[Review]:
    """Parse one PDF. If `stats` (a dict) is given, it is filled with a
    recognised/expected counter: {"file", "expected", "parsed",
    "lost_contexts"} — `expected` counts the "Reviewed in ... on <date>"
    markers (every review carries exactly one), so expected > parsed means
    the parser lost reviews; `lost_contexts` holds text around each lost
    marker for debugging."""
    raw = _extract_text(pdf_path)
    # id_prefix = PDF stem, not `product`: a product's positive/negative
    # export PDFs share the same collapsed `product` name (products.yaml)
    # but are parsed independently, each starting its index at 0 — keying
    # review_id on `product` would let review N of one file collide with
    # review N of the other and silently drop one vote (COUNT DISTINCT).
    reviews = _parse_text(raw, product, stats, id_prefix=pdf_path.stem)
    if stats is not None:
        stats["file"] = pdf_path.name
    return reviews


def _parse_text(raw: str, product: str,
                stats: dict | None = None,
                id_prefix: str | None = None) -> list[Review]:
    text = re.sub(r"\s+", " ", raw)
    id_prefix = product if id_prefix is None else id_prefix

    marks = list(RATING_RE.finditer(text))
    reviews: list[Review] = []
    # author of review i = text between the previous review's "Helpful Report"
    # trailer and the rating marker; for the first review it's the tail of the
    # page header before the first marker.
    pending_author = ""
    first_before = text[:marks[0].start()] if marks else ""
    tm = None
    for t in TRAILER_RE.finditer(first_before):
        tm = t
    pending_author = _clean(first_before[tm.end():]) if tm else \
        " ".join(_clean(first_before).split()[-6:])

    owned_dates: set[int] = set()   # absolute offsets of dates that became reviews
    for i, m in enumerate(marks):
        seg_end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        seg = text[m.end():seg_end]

        author = pending_author
        rating = float(m.group(1).replace(",", "."))

        # cut the segment at the trailer; what's after it is the next author
        t = TRAILER_RE.search(seg)
        if t:
            pending_author = _clean(seg[t.end():])
            seg = seg[:t.start()]
        else:
            pending_author = ""

        dm = DATE_RE.search(seg)
        if not dm:
            continue
        owned_dates.add(m.end() + dm.start())
        title = _clean(seg[:dm.start()])
        rest = seg[dm.end():]
        review_date = _parse_date(dm.group(1))

        variant = ""
        vm = VARIANT_RE.search(rest[:200])
        if vm:
            stop = re.search(r"Verified Purchase|Vine Customer", rest[vm.end():])
            if stop:
                variant = _clean(rest[vm.end():vm.end() + stop.start()])
                rest = rest[vm.end() + stop.start():]
            else:
                first_words = _clean(rest[vm.end():vm.end() + 80])
                variant = " ".join(first_words.split()[:6])
                rest = rest[vm.end() + 80:]
        rest = re.sub(r"^\s*(?:Verified Purchase|Vine Customer Review of Free Product)\s*", "", rest)
        body = _clean(rest)

        reviews.append(Review(
            review_id=f"{id_prefix}:{i}",
            product=product,
            author=author,
            rating=rating,
            title=title,
            date=review_date,
            variant=variant,
            body=body,
        ))
    if stats is not None:
        # every review carries exactly one "Reviewed in ... on <date>" marker,
        # so date markers that ended up owned by no review are lost reviews
        # (a missed rating marker, a glitched text extraction, ...)
        all_dates = list(DATE_RE.finditer(text))
        stats["expected"] = len(all_dates)
        stats["parsed"] = len(reviews)
        stats["lost_contexts"] = [
            _clean(text[max(0, d.start() - 150):d.end() + 250])
            for d in all_dates if d.start() not in owned_dates][:10]
    return reviews


def filter_reviews(reviews: list[Review], max_reviews: int | None = None,
                   cutoff: date | None = None) -> list[Review]:
    """Deduplicates identical (author, date, body) and sorts newest first.
    Optional limits: `max_reviews` (top-N newest) and `cutoff` date —
    by default the whole PDF is analysed."""
    seen: set[tuple] = set()
    unique = []
    for r in reviews:
        key = (r.author, r.date, r.body[:120])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    unique.sort(key=lambda r: r.date or "0000", reverse=True)
    if cutoff:
        unique = [r for r in unique if not r.date or
                  datetime.fromisoformat(r.date).date() >= cutoff]
    if max_reviews:
        unique = unique[:max_reviews]
    return unique
