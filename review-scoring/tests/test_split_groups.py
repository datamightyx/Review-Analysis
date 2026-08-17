"""split_groups: the inverse of consolidate_taxonomy — the only pass that
can pull a distinct theme back OUT of a mega-group the first batch-placement
pass created before the taxonomy's real structure existed."""
import unittest
from pipeline.grouping import _apply_split_result, split_groups
from tests.helpers import tax_with, canon_by_text


def _group_by_name(tax, name):
    return next(g for g in tax.groups.values() if g.name == name)


class FakeSplitLLM:
    """Canned answer for split_groups_async's per-group prompt: always
    proposes splitting off every row whose text contains "litter"."""

    async def json_call_async(self, system, user, schema, max_tokens=16000):
        import re
        ids = re.findall(r'\s(c\d+):', user)
        # pair each id with its row text on the same line
        rows = dict(re.findall(r'(c\d+): "([^"]+)"', user))
        litter_ids = [cid for cid in ids if "litter" in rows.get(cid, "")]
        if len(litter_ids) >= 2:
            return {"splits": [{"name": "Litter Box Use",
                                "canonical_ids": litter_ids}], "moves": []}
        return {"splits": [], "moves": []}


class TestApplySplitResult(unittest.TestCase):
    def _tax(self):
        return tax_with([
            ("Hamster Loves It", "positive", "loves it",
             {"A": ["A:1", "A:2", "A:3"]}),
            ("Hamster Loves It", "positive", "litter box use",
             {"A": ["A:4", "A:5"]}),
            ("Hamster Loves It", "positive", "potty trained fast",
             {"A": ["A:6"]}),
        ])

    def test_split_creates_new_group_and_moves_rows(self):
        tax = self._tax()
        group = _group_by_name(tax, "Hamster Loves It")
        litter = canon_by_text(tax, "litter box use")
        potty = canon_by_text(tax, "potty trained fast")
        result = {"splits": [{"name": "Litter Box Use",
                              "canonical_ids": [litter.id, potty.id]}],
                  "moves": []}
        actions = _apply_split_result(tax, "positive", group, result)
        self.assertEqual(len(actions), 1)
        new_group = _group_by_name(tax, "Litter Box Use")
        self.assertEqual(litter.group_id, new_group.id)
        self.assertEqual(potty.group_id, new_group.id)
        # the original group keeps its remaining row
        remaining = [c for c in tax.canonicals.values()
                    if c.group_id == group.id]
        self.assertEqual([c.text for c in remaining], ["loves it"])

    def test_single_row_split_rejected(self):
        tax = self._tax()
        group = _group_by_name(tax, "Hamster Loves It")
        litter = canon_by_text(tax, "litter box use")
        result = {"splits": [{"name": "Litter Box Use",
                              "canonical_ids": [litter.id]}], "moves": []}
        actions = _apply_split_result(tax, "positive", group, result)
        self.assertEqual(actions, [])
        self.assertEqual(litter.group_id, group.id)   # untouched

    def test_move_into_sibling_group(self):
        tax = self._tax()
        group = _group_by_name(tax, "Hamster Loves It")
        bathing = tax.new_group("positive", "Great for Bathing")
        litter = canon_by_text(tax, "litter box use")
        result = {"splits": [], "moves": [
            {"canonical_id": litter.id, "target_group_id": bathing.id}]}
        actions = _apply_split_result(tax, "positive", group, result)
        self.assertEqual(len(actions), 1)
        self.assertEqual(litter.group_id, bathing.id)

    def test_move_to_wrong_category_rejected(self):
        tax = self._tax()
        group = _group_by_name(tax, "Hamster Loves It")
        other_cat_group = tax.new_group("negative", "Bag Defects")
        litter = canon_by_text(tax, "litter box use")
        result = {"splits": [], "moves": [
            {"canonical_id": litter.id,
             "target_group_id": other_cat_group.id}]}
        actions = _apply_split_result(tax, "positive", group, result)
        self.assertEqual(actions, [])
        self.assertEqual(litter.group_id, group.id)   # untouched


class TestSplitGroupsAsync(unittest.TestCase):
    def test_mega_group_gets_split_end_to_end(self):
        tax = tax_with([
            ("Hamster Loves It", "positive", "loves it",
             {"A": ["A:1", "A:2", "A:3"], "B": ["B:1"]}),
            ("Hamster Loves It", "positive", "litter box use",
             {"A": ["A:4", "A:5"]}),
            ("Hamster Loves It", "positive", "designated it as litter",
             {"A": ["A:6"]}),
        ])
        actions = split_groups(tax, FakeSplitLLM(),
                               min_share=0.0, min_rows=2)
        self.assertTrue(any("Litter Box Use" in a for a in actions))
        new_group = _group_by_name(tax, "Litter Box Use")
        self.assertEqual(len(new_group.canonicals(tax)), 2)


if __name__ == "__main__":
    unittest.main()
