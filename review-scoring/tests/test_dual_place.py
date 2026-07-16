"""Within-group dual placement: one review naming two benefits that each have
their own row counts in BOTH rows (the emergency/first-aid case)."""
import json
import tempfile
import unittest
from pathlib import Path
from pipeline.grouping import apply_overrides, _apply_assignment
from tests.helpers import tax_with, canon_by_text


class TestDualPlaceOverride(unittest.TestCase):
    def _tax(self):
        return tax_with([
            ("Must have", "positive", "emergencies row",
             {"A": ["A:1"]}),                    # holds the quote
            ("Must have", "positive", "first aid row",
             {"A": ["A:2"]}),
        ])

    def test_dual_place_adds_to_second_row_deduped(self):
        tax = self._tax()
        # put the dual quote on the emergencies row
        canon_by_text(tax, "emergencies row").quotes["A"] = ["great for an emergency or first aid kit"]
        canon_by_text(tax, "emergencies row").review_ids["A"] = ["A:1"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "overrides.json"
            p.write_text(json.dumps({"dual_place": [
                {"quote": "great for an emergency or first aid kit",
                 "rows": ["emergencies row", "first aid row"]}]}), encoding="utf-8")
            apply_overrides(tax, p)
            apply_overrides(tax, p)   # idempotent

        first = canon_by_text(tax, "first aid row")
        self.assertIn("A:1", first.review_ids["A"])   # gained the review
        self.assertEqual(first.votes["A"], 2)          # A:2 + A:1, not more on rerun
        self.assertIn("A:1", canon_by_text(tax, "emergencies row").review_ids["A"])


class TestSecondCanonicalAssignment(unittest.TestCase):
    def test_second_canonical_id_counts_in_both_rows(self):
        tax = tax_with([
            ("Must have", "positive", "emergencies row", {"A": ["A:1"]}),
            ("Must have", "positive", "first aid row", {"A": ["A:2"]}),
        ])
        em = canon_by_text(tax, "emergencies row")
        fa = canon_by_text(tax, "first aid row")
        b = {"text": "emergencies row", "gist": "",
             "counts": {"B": 1}, "raws": {"B": ["x"]},
             "review_ids": {"B": ["B:5"]}}
        a = {"canonical_id": em.id, "group_id": em.group_id,
             "second_canonical_id": fa.id}
        _apply_assignment(tax, "positive", b, a, {})
        self.assertEqual(em.votes.get("B"), 1)
        self.assertEqual(fa.votes.get("B"), 1)
        self.assertIn("B:5", fa.review_ids["B"])


if __name__ == "__main__":
    unittest.main()
