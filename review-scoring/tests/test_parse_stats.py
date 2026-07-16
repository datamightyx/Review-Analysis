"""Recognised/expected counter of the PDF parser (_parse_text stats).

`expected` counts "Reviewed in ... on <date>" markers — every review carries
exactly one, so expected > parsed means the parser lost reviews (a missed
rating marker, glitched text extraction), and lost_contexts shows where.
"""
import pytest

from pipeline.pdf_parser import _parse_text


def _review_block(author, title, body, date="January 5, 2026", rating="5.0"):
    return (f"{author} {rating} out of 5 stars {title} "
            f"Reviewed in the United States on {date} Verified Purchase "
            f"{body} Helpful Report")


HEADER = "Amazon.com Customer reviews 4.3 out of 5 stars 120 global ratings "


def test_all_reviews_parsed_no_loss():
    text = HEADER + _review_block("Ann", "Great", "Works great.") + " " + \
        _review_block("Bob", "Meh", "Did not stick.", date="March 2, 2026",
                      rating="2.0")
    stats = {}
    reviews = _parse_text(text, "P", stats)
    assert len(reviews) == 2
    assert stats["expected"] == 2
    assert stats["parsed"] == 2
    assert stats["lost_contexts"] == []


def test_lost_review_detected():
    # the second review's rating marker is corrupted ("ou t of") so its
    # segment fuses into the first review — but its date marker survives
    lost = _review_block("Bob", "Meh", "Did not stick.",
                         date="March 2, 2026", rating="2.0")
    lost = lost.replace("out of 5 stars", "ou t of 5 stars")
    text = HEADER + _review_block("Ann", "Great", "Works great.") + " " + lost
    stats = {}
    reviews = _parse_text(text, "P", stats)
    assert len(reviews) == 1
    assert stats["expected"] == 2
    assert stats["parsed"] == 1
    assert len(stats["lost_contexts"]) == 1
    assert "March 2, 2026" in stats["lost_contexts"][0]


def test_stats_optional():
    text = HEADER + _review_block("Ann", "Great", "Works great.")
    assert len(_parse_text(text, "P")) == 1
