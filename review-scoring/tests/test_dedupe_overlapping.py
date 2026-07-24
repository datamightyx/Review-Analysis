"""Intra-review overlap dedup: a short clause + the longer sentence quoting
it must not both survive (that would count one review twice)."""
import unittest
from pipeline.extract import dedupe_overlapping
from tests.helpers import phrase


def run(quotes, **kw):
    res, log = dedupe_overlapping([phrase(q, **kw) for q in quotes])
    return [p.quote for p in res], log


class TestDedupeOverlapping(unittest.TestCase):
    def test_enumeration_or_is_split(self):
        out, log = run(["Not suitable for kids under at least 10",
                        "Not suitable for kids under at least 10, or any sort of facial cuts"])
        self.assertIn("Not suitable for kids under at least 10", out)
        self.assertIn("any sort of facial cuts", out)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(log["split"]), 1)

    def test_sentence_boundary_is_split(self):
        out, _ = run(["The tape was old", "The tape was old. Nothing would stick"])
        self.assertEqual(sorted(out), sorted(["The tape was old", "Nothing would stick"]))

    def test_verbose_infinitive_is_dropped(self):
        out, log = run(["Great variety", "Great variety to take care of multiple wounds"],
                       category="positive")
        self.assertEqual(out, ["Great variety"])
        self.assertEqual(len(log["dropped"]), 1)

    def test_comma_less_and_is_dropped(self):
        out, _ = run(["Gauze roll is nice quality",
                      "Gauze roll is nice quality and feels real spongy"],
                     category="positive")
        self.assertEqual(out, ["Gauze roll is nice quality"])

    def test_no_overlap_untouched(self):
        out, log = run(["Not suitable for kids under at least 10", "any sort of facial cuts"])
        self.assertEqual(len(out), 2)
        self.assertFalse(log["split"] or log["dropped"])

    def test_different_reviews_untouched(self):
        res, log = dedupe_overlapping([
            phrase("Great variety", "positive", "P:1"),
            phrase("Great variety to take care of multiple wounds", "positive", "P:2"),
        ])
        self.assertEqual(len(res), 2)
        self.assertFalse(log["split"] or log["dropped"])

    def test_different_category_untouched(self):
        # same review, one clause tagged negative, superset tagged usage -> not an overlap
        res, log = dedupe_overlapping([
            phrase("cut on face", "usage", "P:1"),
            phrase("cut on face and it stung", "negative", "P:1"),
        ])
        self.assertEqual(len(res), 2)

    def test_split_off_tail_is_rechecked_against_a_third_phrase(self):
        # The long phrase splits into a prefix ("...camping trips and
        # hiking") plus a distinct tail ("perfect for the beach and pool").
        # A third phrase ("the beach and pool") is a strict SUBSET of that
        # split-off tail — it must absorb/drop the (now redundant) tail, or
        # both survive and double-count this one review.
        out, log = run([
            "Great for camping trips and hiking, plus perfect for the beach and pool",
            "Great for camping trips and hiking",
            "the beach and pool",
        ])
        self.assertEqual(len(out), 2)
        self.assertIn("Great for camping trips and hiking", out)
        self.assertIn("the beach and pool", out)
        self.assertNotIn("perfect for the beach and pool", out)


if __name__ == "__main__":
    unittest.main()
