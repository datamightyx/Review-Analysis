"""Labels fold by grammatical number; verbatim wordings never do.

A relation row ("chinchilla" / "chinchillas"), a wish gist ("bigger amount" /
"bigger amounts") and a group name are LABELS: singular and plural are one
thing. Relation categories are ungated, so nothing used to visit those rows
at all — 20 rows under "Other Small Pets" for 14 actual species in the
Hamster Sand workbook (2026-08-17).
"""
import unittest
from collections import defaultdict

from pipeline.grouping import (_deterministic_prepass, _group_named,
                               dedupe_group_names, fold_relation_rows,
                               reconcile_votes)
from pipeline.similarity import fold_label
from helpers import tax_with


def uniq(text, category, gist="", **votes):
    """One bucket in the shape _unique_phrases returns."""
    b = {"text": text, "category": category, "counts": defaultdict(int),
         "raws": defaultdict(list), "review_ids": defaultdict(list),
         "pairs": defaultdict(list), "relation": "", "gist": gist,
         "quote_en": ""}
    for product, rids in votes.items():
        b["counts"][product] = len(rids)
        b["raws"][product] = [text]
        b["review_ids"][product] = list(rids)
        b["pairs"][product] = [(r, text) for r in rids]
    return b


class TestFoldLabel(unittest.TestCase):
    def test_regular_plurals_fold(self):
        for plural, singular in [("gerbils", "gerbil"),
                                 ("chinchillas", "chinchilla"),
                                 ("quails", "quail"),
                                 ("hedgehogs", "hedgehog"),
                                 ("small pets", "small pet"),
                                 ("bigger amounts", "bigger amount")]:
            self.assertEqual(fold_label(plural), fold_label(singular), plural)

    def test_es_and_ies_endings(self):
        self.assertEqual(fold_label("boxes"), fold_label("box"))
        self.assertEqual(fold_label("puppies"), fold_label("puppy"))

    def test_double_s_and_short_tokens_survive(self):
        self.assertEqual(fold_label("glass"), "glass")
        self.assertEqual(fold_label("less"), "less")
        self.assertEqual(fold_label("as is"), "as is")

    def test_irregular_plurals_are_left_alone(self):
        """A missed merge costs a row; a wrong one corrupts a count."""
        self.assertNotEqual(fold_label("mice"), fold_label("mouse"))

    def test_different_labels_do_not_collide(self):
        self.assertNotEqual(fold_label("gerbils"),
                            fold_label("hamster and gerbil"))
        self.assertNotEqual(fold_label("gerbils"),
                            fold_label("gerbils, mice, small pets"))


class TestFoldRelationRows(unittest.TestCase):
    def test_number_variants_of_one_species_merge(self):
        tax = tax_with([
            ("Other Small Pets", "usage", "chinchilla", {"JFWOD": ["r1", "r2"]}),
            ("Other Small Pets", "usage", "chinchillas", {"JFWOD": ["r3"]}),
            ("Other Small Pets", "usage", "hedgehog", {"DR_DUDU": ["r4"]}),
        ])

        actions = fold_relation_rows(tax)

        g = _group_named(tax, "usage", "Other Small Pets")
        texts = {c.text for c in g.canonicals(tax)}
        self.assertEqual(len(actions), 1)
        self.assertEqual(texts, {"chinchilla", "hedgehog"})  # top wording wins
        self.assertEqual(g.total(tax), 4)                    # no vote lost

    def test_shared_review_counts_once(self):
        tax = tax_with([
            ("Gerbils", "usage", "gerbil", {"A": ["r1"]}),
            ("Gerbils", "usage", "gerbils", {"A": ["r1"]}),   # same review
        ])
        fold_relation_rows(tax)
        reconcile_votes(tax)
        self.assertEqual(_group_named(tax, "usage", "Gerbils").total(tax), 1)

    def test_never_merges_across_groups(self):
        """The same label in two groups is a deliberate dual placement."""
        tax = tax_with([
            ("Hamsters", "usage", "hamster", {"A": ["r1"]}),
            ("Sand Bathing", "usage", "hamsters", {"A": ["r2"]}),
        ])
        self.assertEqual(fold_relation_rows(tax), [])
        self.assertEqual(len(tax.canonicals), 2)

    def test_verbatim_categories_are_never_folded(self):
        """positive/negative rows are customer wordings — the exact phrasing
        is the product of the analysis."""
        tax = tax_with([
            ("Nice Color", "positive", "pretty flower petal", {"A": ["r1"]}),
            ("Nice Color", "positive", "pretty flower petals", {"A": ["r2"]}),
        ])
        self.assertEqual(fold_relation_rows(tax), [])
        self.assertEqual(len(tax.canonicals), 2)


class TestFoldedGroupNames(unittest.TestCase):
    def test_number_variant_group_names_merge(self):
        tax = tax_with([
            ("Chinchillas", "usage", "chinchilla", {"A": ["r1", "r2"]}),
            ("Chinchilla", "usage", "chinchilla pet", {"A": ["r3"]}),
        ])

        dedupe_group_names(tax)

        groups = tax.groups_for("usage")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].name, "Chinchillas")   # the stronger one
        self.assertEqual(groups[0].total(tax), 3)

    def test_group_lookup_is_number_insensitive(self):
        tax = tax_with([("Bigger amount", "improvement", "wish it was bigger",
                         {"A": ["r1"]})])
        self.assertIsNotNone(_group_named(tax, "improvement", "bigger amounts"))


class TestGistClustering(unittest.TestCase):
    def test_wish_joins_the_existing_theme_without_a_judge(self):
        tax = tax_with([("Bigger amount", "improvement",
                         "I wish it was bigger", {"Sukh": ["r1"]})])
        b = uniq("just wish it came with a larger quantity", "improvement",
                 gist="Bigger amounts", Sukh=["r2"])

        pending = _deterministic_prepass(tax, "improvement", [b])

        self.assertEqual(pending, [])            # never reached the judge
        g = _group_named(tax, "improvement", "Bigger amount")
        self.assertEqual(len(tax.groups_for("improvement")), 1)
        # the row keeps the customer's verbatim wording — only the GROUP
        # came from the gist
        self.assertIn("just wish it came with a larger quantity",
                      {c.text for c in g.canonicals(tax)})
        self.assertEqual(g.total(tax), 2)

    def test_unknown_gist_still_goes_to_the_judge(self):
        tax = tax_with([("Bigger amount", "improvement",
                         "I wish it was bigger", {"Sukh": ["r1"]})])
        b = uniq("please make a silica free version", "improvement",
                 gist="Silica free sand", Sukh=["r2"])

        self.assertEqual(_deterministic_prepass(tax, "improvement", [b]), [b])

    def test_gist_is_ignored_outside_gist_named_categories(self):
        tax = tax_with([("Nice Color", "positive", "pretty sand",
                         {"A": ["r1"]})])
        b = uniq("lovely shade", "positive", gist="Nice Color", A=["r2"])

        self.assertEqual(_deterministic_prepass(tax, "positive", [b]), [b])


if __name__ == "__main__":
    unittest.main()
