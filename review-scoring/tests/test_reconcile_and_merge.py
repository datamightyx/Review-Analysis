"""One review = one vote: reconcile_votes heals counter-style inflation, and
merging two rows that share a review must not double-count it."""
import unittest
from pipeline.grouping import reconcile_votes, _merge_canonical_into
from tests.helpers import tax_with, canon_by_text


class TestReconcile(unittest.TestCase):
    def test_reconcile_clamps_to_distinct_reviews(self):
        tax = tax_with([("G", "positive", "works", {"A": ["A:1", "A:2"]})])
        c = canon_by_text(tax, "works")
        c.votes["A"] = 9                       # inflated counter (only 2 ids)
        removed = reconcile_votes(tax)
        self.assertEqual(c.votes["A"], 2)
        self.assertEqual(removed, 7)

    def test_reconcile_ignores_idless_cells(self):
        tax = tax_with([("G", "positive", "works", {"A": ["A:1"]})])
        c = canon_by_text(tax, "works")
        c.review_ids["B"] = []                 # legacy: votes but no ids
        c.votes["B"] = 3
        reconcile_votes(tax)
        self.assertEqual(c.votes["B"], 3)      # untouched

    def test_merge_shared_review_not_double_counted(self):
        tax = tax_with([
            ("G", "positive", "keep", {"A": ["A:1", "A:2"]}),
            ("G", "positive", "other", {"A": ["A:2", "A:3"]}),   # shares A:2
        ])
        keep = canon_by_text(tax, "keep")
        other = canon_by_text(tax, "other")
        _merge_canonical_into(tax, keep, other)
        # union of ids = {A:1, A:2, A:3} -> 3, NOT 2+2=4
        self.assertEqual(keep.votes["A"], 3)
        self.assertEqual(sorted(keep.review_ids["A"]), ["A:1", "A:2", "A:3"])


if __name__ == "__main__":
    unittest.main()
