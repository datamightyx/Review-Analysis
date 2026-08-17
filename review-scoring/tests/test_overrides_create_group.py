"""apply_overrides: move_canonical must auto-create a missing target group
(previously it silently no-opped, so a manual fix to a mega-group had no way
to survive the next run), scoped to the moved row's OWN category; create_group
is the general hand-carve-a-new-USP tool that survives reruns the same way."""
import json
import tempfile
import unittest
from pathlib import Path
from pipeline.grouping import apply_overrides
from tests.helpers import tax_with, canon_by_text


def _group_by_name(tax, name):
    return next((g for g in tax.groups.values() if g.name == name), None)


class TestMoveCanonicalAutoCreates(unittest.TestCase):
    def test_missing_target_group_is_created(self):
        tax = tax_with([
            ("Hamster Loves It", "positive", "litter box use",
             {"A": ["A:1"]}),
        ])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "overrides.json"
            p.write_text(json.dumps({"move_canonical": {
                "litter box use": "Litter Box Use"}}), encoding="utf-8")
            apply_overrides(tax, p)
        target = _group_by_name(tax, "Litter Box Use")
        self.assertIsNotNone(target)
        litter = canon_by_text(tax, "litter box use")
        self.assertEqual(litter.group_id, target.id)

    def test_scoped_to_own_category_not_cross_category_namesake(self):
        tax = tax_with([
            ("Bag Defects", "negative", "torn on arrival", {"A": ["A:1"]}),
            ("Positive Bag Defects", "positive", "sturdy bag", {"A": ["A:2"]}),
        ])
        # a same-named group in the WRONG category must never absorb this row
        neg_group = _group_by_name(tax, "Bag Defects")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "overrides.json"
            p.write_text(json.dumps({"move_canonical": {
                "sturdy bag": "Bag Defects"}}), encoding="utf-8")
            apply_overrides(tax, p)
        moved = canon_by_text(tax, "sturdy bag")
        new_group = tax.groups[moved.group_id]
        self.assertEqual(new_group.category, "positive")
        self.assertNotEqual(new_group.id, neg_group.id)

    def test_rerun_is_idempotent(self):
        tax = tax_with([
            ("Hamster Loves It", "positive", "litter box use",
             {"A": ["A:1"]}),
        ])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "overrides.json"
            p.write_text(json.dumps({"move_canonical": {
                "litter box use": "Litter Box Use"}}), encoding="utf-8")
            apply_overrides(tax, p)
            apply_overrides(tax, p)   # second run must not create a duplicate
        groups = [g for g in tax.groups.values() if g.name == "Litter Box Use"]
        self.assertEqual(len(groups), 1)


class TestCreateGroup(unittest.TestCase):
    def test_carves_rows_out_of_existing_group_by_text(self):
        tax = tax_with([
            ("Hamster Loves It", "positive", "loves it", {"A": ["A:1"]}),
            ("Hamster Loves It", "positive", "designated it as litter box",
             {"A": ["A:2"]}),
        ])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "overrides.json"
            p.write_text(json.dumps({"create_group": [
                {"category": "positive", "name": "Litter Box Use",
                 "rows": ["designated it as litter box"]}]}),
                encoding="utf-8")
            apply_overrides(tax, p)
        target = _group_by_name(tax, "Litter Box Use")
        self.assertIsNotNone(target)
        moved = canon_by_text(tax, "designated it as litter box")
        self.assertEqual(moved.group_id, target.id)
        stayed = canon_by_text(tax, "loves it")
        self.assertNotEqual(stayed.group_id, target.id)

    def test_reuses_existing_same_named_group(self):
        tax = tax_with([
            ("Litter Box Use", "positive", "already here", {"A": ["A:1"]}),
            ("Hamster Loves It", "positive", "designated it as litter box",
             {"A": ["A:2"]}),
        ])
        existing = _group_by_name(tax, "Litter Box Use")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "overrides.json"
            p.write_text(json.dumps({"create_group": [
                {"category": "positive", "name": "Litter Box Use",
                 "rows": ["designated it as litter box"]}]}),
                encoding="utf-8")
            apply_overrides(tax, p)
        groups = [g for g in tax.groups.values() if g.name == "Litter Box Use"]
        self.assertEqual(len(groups), 1)   # no duplicate group created
        self.assertEqual(groups[0].id, existing.id)


if __name__ == "__main__":
    unittest.main()
