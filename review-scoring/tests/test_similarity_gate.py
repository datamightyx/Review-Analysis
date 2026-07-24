"""Merge gate: praise tiers and qualifier families never collapse across the
SOP boundaries (locks the hard-won rules in memory)."""
import unittest
from pipeline.similarity import merge_blocked, merge_compatible


class TestPraiseTiers(unittest.TestCase):
    def test_plain_vs_strong_blocked(self):
        # "works well" (plain) must NOT merge into "works great" (strong)
        self.assertIsNotNone(merge_blocked("Works well", "Works great"))
        self.assertFalse(merge_compatible("Works well", "Works great"))

    def test_same_tier_allowed(self):
        # "works good" and "works well" are both plain tier
        self.assertIsNone(merge_blocked("works good", "works well"))

    def test_recognized_tier_vs_unlisted_word_blocked(self):
        # "well" is tier 0; "nicely" isn't in any tier list at all, so the
        # tier-mismatch check above never even sees a conflict (empty set on
        # one side) — this must not read as "no rule violated"
        self.assertIsNotNone(merge_blocked(
            "Wraps well.", "It is easy to use and wraps nicely."))


class TestQualifierFamilies(unittest.TestCase):
    def test_fast_vs_quickly_separate(self):
        # different qualifier wording stays separate canonicals
        self.assertFalse(merge_compatible("stops bleeding fast",
                                          "stops bleeding quickly"))


class TestNegation(unittest.TestCase):
    def test_one_sided_negation_blocked(self):
        self.assertIsNotNone(merge_blocked("doesn't stick", "sticks well"))


if __name__ == "__main__":
    unittest.main()
