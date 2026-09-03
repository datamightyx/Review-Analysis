"""One theme = one group.

Two groups carrying the same name in one category are the same theme: the
workbook renders them as two blocks under one label, each with its own
subtotal. The split pass must reuse an existing name instead of creating the
collision, dedupe_group_names heals a taxonomy that already has one, and
quality.workbook_warnings tells the human it happened.
"""
import unittest

from pipeline.grouping import (_apply_split_result, _group_named,
                               dedupe_group_names, reconcile_votes)
from pipeline.quality import workbook_warnings
from helpers import tax_with, canon_by_text


class TestDedupeGroupNames(unittest.TestCase):
    def test_same_named_groups_merge_and_keep_every_vote(self):
        tax = tax_with([
            ("Litter Box / Potty Use", "positive", "uses it as a potty",
             {"A": ["r1", "r2"]}),
            ("Litter Box / Potty Use", "positive", "designated it his toilet",
             {"A": ["r3"]}),
        ])
        # helpers builds one group per (name, category) pair, so force the
        # collision the split pass used to produce
        second = tax.new_group("positive", "Litter Box / Potty Use")
        canon_by_text(tax, "designated it his toilet").group_id = second.id
        self.assertEqual(len(tax.groups_for("positive")), 2)

        actions = dedupe_group_names(tax)

        self.assertEqual(len(actions), 1)
        groups = tax.groups_for("positive")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].name, "Litter Box / Potty Use")
        self.assertEqual({c.text for c in groups[0].canonicals(tax)},
                         {"uses it as a potty", "designated it his toilet"})
        self.assertEqual(groups[0].total(tax), 3)

    def test_strongest_group_survives(self):
        tax = tax_with([("Too Dusty", "negative", "so much dust",
                         {"A": ["r1", "r2", "r3"]})])
        weak = tax.new_group("negative", "too dusty")   # differs only in case
        c = tax.new_canonical("a bit dusty", weak.id)
        c.votes["A"] = 1
        c.review_ids["A"] = ["r9"]

        dedupe_group_names(tax)

        groups = tax.groups_for("negative")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].name, "Too Dusty")   # not "too dusty"
        self.assertEqual(groups[0].total(tax), 4)

    def test_shared_review_is_not_double_counted_after_merge(self):
        """The same review sitting in both copies is one vote, not two."""
        tax = tax_with([("Nice Color", "positive", "pretty sand",
                         {"A": ["r1"]})])
        twin = tax.new_group("positive", "Nice Color")
        c = tax.new_canonical("pretty sand", twin.id)
        c.votes["A"] = 1
        c.review_ids["A"] = ["r1"]          # same review id

        dedupe_group_names(tax)
        reconcile_votes(tax)                # the write path's order

        groups = tax.groups_for("positive")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].total(tax), 1)

    def test_untouched_when_names_differ(self):
        tax = tax_with([
            ("Too Dusty", "negative", "so much dust", {"A": ["r1"]}),
            ("Poor Clumping", "negative", "never clumps", {"A": ["r2"]}),
        ])
        self.assertEqual(dedupe_group_names(tax), [])
        self.assertEqual(len(tax.groups_for("negative")), 2)

    def test_split_reuses_existing_group_instead_of_colliding(self):
        tax = tax_with([
            ("Hamster Loves It", "positive", "uses it as a potty",
             {"A": ["r1", "r2"]}),
            ("Hamster Loves It", "positive", "made it his toilet",
             {"A": ["r3", "r4"]}),
            ("Litter Box / Potty Use", "positive", "great litter box",
             {"A": ["r5"]}),
        ])
        big = _group_named(tax, "positive", "Hamster Loves It")
        result = {"splits": [{
            "name": "Litter Box / Potty Use",
            "canonical_ids": [c.id for c in big.canonicals(tax)],
        }]}

        _apply_split_result(tax, "positive", big, result)

        names = [g.name for g in tax.groups_for("positive")]
        self.assertEqual(names.count("Litter Box / Potty Use"), 1)
        target = _group_named(tax, "positive", "Litter Box / Potty Use")
        self.assertEqual(len(target.canonicals(tax)), 3)

    def test_split_into_its_own_name_is_a_no_op(self):
        tax = tax_with([
            ("Too Dusty", "negative", "so much dust", {"A": ["r1", "r2"]}),
            ("Too Dusty", "negative", "dust everywhere", {"A": ["r3", "r4"]}),
        ])
        g = _group_named(tax, "negative", "Too Dusty")
        result = {"splits": [{"name": "too dusty",
                              "canonical_ids": [c.id for c in g.canonicals(tax)]}]}

        _apply_split_result(tax, "negative", g, result)

        self.assertEqual(len(tax.groups_for("negative")), 1)
        self.assertEqual(len(g.canonicals(tax)), 2)

    def test_quality_warns_before_the_merge_happens(self):
        tax = tax_with([("Nice Color", "positive", "pretty sand",
                         {"A": ["r1"]})])
        twin = tax.new_group("positive", "Nice Color")
        c = tax.new_canonical("lovely shade", twin.id)
        c.votes["A"] = 1
        c.review_ids["A"] = ["r2"]

        hits = [w for w in workbook_warnings(tax) if "однією назвою" in w]

        self.assertEqual(len(hits), 1)          # one line per NAME, not per group
        self.assertIn("Nice Color", hits[0])

    def test_quality_warns_about_number_variants_too(self):
        """The warning is keyed exactly like the merge it announces — a
        number variant must not be merged silently."""
        tax = tax_with([
            ("Chinchillas", "usage", "chinchilla", {"A": ["r1"]}),
            ("Chinchilla", "usage", "chinchilla pet", {"A": ["r2"]}),
        ])

        hits = [w for w in workbook_warnings(tax) if "однією назвою" in w]

        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
