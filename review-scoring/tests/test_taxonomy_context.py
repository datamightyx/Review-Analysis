"""_taxonomy_context(max_canonicals=None) must show every row — the fix for
consolidate_taxonomy's old blind spot, where a `moves` proposal could only
ever target a row the prompt actually showed, and the default cutoff (30)
silently exempted oversized groups from cleanup."""
import unittest
from pipeline.grouping import _taxonomy_context
from tests.helpers import tax_with


class TestTaxonomyContextTruncation(unittest.TestCase):
    def _tax(self, n):
        rows = [("Big Group", "positive", f"row {i}", {"A": [f"A:{i}"]})
               for i in range(n)]
        return tax_with(rows)

    def test_default_truncates_long_tail(self):
        tax = self._tax(20)
        ctx = _taxonomy_context(tax, "positive", max_canonicals=5)
        self.assertIn("… and 15 more rows", ctx)
        self.assertNotIn("row 19", ctx)

    def test_none_shows_every_row(self):
        tax = self._tax(20)
        ctx = _taxonomy_context(tax, "positive", max_canonicals=None)
        self.assertNotIn("more rows", ctx)
        for i in range(20):
            self.assertIn(f"row {i}", ctx)


if __name__ == "__main__":
    unittest.main()
