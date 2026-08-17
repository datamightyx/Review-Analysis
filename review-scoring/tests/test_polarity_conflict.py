"""Polarity veto: two rows about the same tracked attribute but with
opposite claims ("unscented" vs "smells amazing") must never merge, and the
check is a no-op when no attribute_families are configured (universal-tool
default)."""
import unittest
from pipeline.similarity import (polarity_conflict, set_attribute_families,
                                 merge_blocked)


class TestPolarityConflict(unittest.TestCase):
    def setUp(self):
        set_attribute_families([
            ["scent", "smell", "odor", "fragrance", "aroma"],
            ["dust", "dusty"],
            ["clump", "clumping"],
        ])

    def tearDown(self):
        set_attribute_families(None)   # never leak into other tests

    def test_negation_word_conflicts_with_assertion(self):
        self.assertTrue(polarity_conflict(
            "not dusty at all", "very dusty and messy"))

    def test_free_suffix_conflicts_with_assertion(self):
        self.assertTrue(polarity_conflict(
            "completely dust free", "a huge cloud of dust"))

    def test_un_prefix_conflicts_with_assertion(self):
        self.assertTrue(polarity_conflict(
            "unscented and clean", "a pleasant floral scent"))

    def test_same_polarity_no_conflict(self):
        self.assertFalse(polarity_conflict(
            "unscented and clean", "no odor at all"))

    def test_unrelated_attribute_no_conflict(self):
        self.assertFalse(polarity_conflict(
            "unscented and clean", "very soft and fine"))

    def test_noop_without_configured_families(self):
        set_attribute_families(None)
        self.assertFalse(polarity_conflict(
            "not dusty at all", "very dusty and messy"))

    def test_merge_blocked_reports_polarity_conflict(self):
        reason = merge_blocked("no scent at all", "smells absolutely amazing")
        self.assertIsNotNone(reason)
        self.assertIn("полярність", reason)


if __name__ == "__main__":
    unittest.main()
