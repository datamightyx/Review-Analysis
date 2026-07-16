"""Product identity: a product's positive & negative PDFs collapse to ONE key
/ column instead of two pseudo-products."""
import unittest
from pipeline.models import product_key
from tests.helpers import tax_with, canon_by_text


class TestProductKey(unittest.TestCase):
    def test_strips_sentiment_suffixes(self):
        cases = {
            "B0FFT5KR8L_positive": "B0FFT5KR8L",
            "B0FFT5KR8L_negative": "B0FFT5KR8L",
            "B0FB86T1PJ_ppositive": "B0FB86T1PJ",
            "B0D4PHX83N_positive +negative": "B0D4PHX83N",
        }
        for stem, want in cases.items():
            self.assertEqual(product_key(stem), want, stem)

    def test_plain_name_untouched(self):
        self.assertEqual(product_key("Acme First Aid Kit"), "Acme First Aid Kit")
        # a product legitimately containing 'positive' as a word is not split
        self.assertEqual(product_key("PositiveVibes"), "PositiveVibes")


class TestRemapProducts(unittest.TestCase):
    def test_pos_neg_merge_into_one_column(self):
        tax = tax_with([
            ("Kit", "usage", "first aid kit",
             {"B0X_positive": ["B0X_positive:1", "B0X_positive:2"],
              "B0X_negative": ["B0X_negative:9"]}),
        ])
        renamed = tax.remap_products()
        c = canon_by_text(tax, "first aid kit")
        self.assertEqual(set(c.votes), {"B0X"})
        self.assertEqual(renamed, 2)
        # union of ids preserved, votes summed (reconcile later -> 3 distinct)
        self.assertEqual(sorted(c.review_ids["B0X"]),
                         ["B0X_negative:9", "B0X_positive:1", "B0X_positive:2"])
        self.assertEqual(c.votes["B0X"], 3)


if __name__ == "__main__":
    unittest.main()
