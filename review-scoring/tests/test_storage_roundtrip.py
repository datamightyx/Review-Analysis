"""SQLite storage layer: save/load round-trip and its invariants."""
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from pipeline.models import Taxonomy
from storage.db_client import DB
from tests.helpers import tax_with, canon_by_text


@contextmanager
def temp_db():
    """Fresh DB in a temp dir; closed before cleanup (Windows can't delete
    an open sqlite file)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = DB(Path(tmp) / "scoring.db")
        try:
            yield db
        finally:
            db.close()


class TestTaxonomyRoundTrip(unittest.TestCase):
    def test_structure_votes_quotes_ids_survive(self):
        tax = tax_with([
            ("Stops bleeding", "positive", "stopped the bleeding fast",
             {"A": ["A:1", "A:2"], "B": ["B:1"]}),
            ("Stops bleeding", "positive", "bleeding stopped instantly",
             {"A": ["A:3"]}),
            ("Hard to apply", "negative", "powder goes everywhere",
             {"B": ["B:2", "B:3"]}),
        ])
        c = canon_by_text(tax, "stopped the bleeding fast")
        c.quotes["A"] = ["stopped the bleeding fast", "bleeding stopped quick"]
        tax._next_id = 42
        with temp_db() as db:
            db.save_taxonomy(tax)
            back = db.load_taxonomy()
        self.assertEqual(back._next_id, 42)
        self.assertEqual(list(back.groups), list(tax.groups))
        self.assertEqual(list(back.canonicals), list(tax.canonicals))
        for cid, orig in tax.canonicals.items():
            got = back.canonicals[cid]
            self.assertEqual(got.text, orig.text)
            self.assertEqual(got.group_id, orig.group_id)
            self.assertEqual(got.votes, orig.votes)
            self.assertEqual(got.quotes, orig.quotes)
            self.assertEqual(got.review_ids, orig.review_ids)
        g = next(iter(back.groups.values()))
        self.assertEqual(g.name, "Stops bleeding")
        self.assertEqual(g.category, "positive")

    def test_votes_are_count_distinct_review_ids(self):
        """Phantom votes (count > distinct ids) heal on load — the SQL
        aggregate IS reconcile_votes."""
        tax = tax_with([
            ("G", "positive", "great stuff", {"A": ["A:1", "A:2"]}),
        ])
        c = canon_by_text(tax, "great stuff")
        c.review_ids["A"] = ["A:1", "A:2", "A:1"]   # duplicate id
        c.votes["A"] = 5                             # phantom count
        with temp_db() as db:
            db.save_taxonomy(tax)
            back = db.load_taxonomy()
        self.assertEqual(back.canonicals[c.id].votes["A"], 2)
        self.assertEqual(back.canonicals[c.id].review_ids["A"], ["A:1", "A:2"])

    def test_dual_placement_one_quote_two_rows(self):
        """The same (product, review_id, quote) under two canonicals is two
        legal rows — UNIQUE includes canonical_id."""
        tax = tax_with([
            ("G", "positive", "row A", {"P": ["P:7"]}),
            ("G", "positive", "row B", {"P": ["P:7"]}),
        ])
        for text in ("row A", "row B"):
            canon_by_text(tax, text).quotes["P"] = ["works in an emergency"]
        with temp_db() as db:
            db.save_taxonomy(tax)
            back = db.load_taxonomy()
        for text in ("row A", "row B"):
            got = canon_by_text(back, text)
            self.assertEqual(got.votes["P"], 1)
            self.assertEqual(got.quotes["P"], ["works in an emergency"])

    def test_legacy_votes_without_review_ids_survive(self):
        tax = tax_with([("G", "positive", "old row", {})])
        c = canon_by_text(tax, "old row")
        c.votes["A"] = 3   # count only, no ids (pre-review_ids data)
        with temp_db() as db:
            db.save_taxonomy(tax)
            back = db.load_taxonomy()
        got = canon_by_text(back, "old row")
        self.assertEqual(got.votes["A"], 3)
        self.assertEqual(got.review_ids, {})   # synthesized ids filtered out

    def test_rerun_clears_and_rebuilds_votes(self):
        tax = tax_with([("G", "positive", "row", {"A": ["A:1", "A:2"]})])
        with temp_db() as db:
            db.save_taxonomy(tax)
            c = canon_by_text(tax, "row")
            c.review_ids["A"] = ["A:9"]         # new run: different reviews
            c.votes["A"] = 1
            db.save_taxonomy(tax)               # full rewrite
            back = db.load_taxonomy()
        got = canon_by_text(back, "row")
        self.assertEqual(got.votes["A"], 1)
        self.assertEqual(got.review_ids["A"], ["A:9"])

    def test_empty_db_loads_empty_taxonomy(self):
        with temp_db() as db:
            back = db.load_taxonomy()
        self.assertIsInstance(back, Taxonomy)
        self.assertEqual(back.groups, {})
        self.assertEqual(back.canonicals, {})
        self.assertEqual(back._next_id, 1)


class TestLLMCache(unittest.TestCase):
    def test_get_put_and_insert_or_ignore(self):
        with temp_db() as db:
            self.assertIsNone(db.llm_cache_get("k1"))
            db.llm_cache_put("k1", "anthropic", "m", "medium", {"a": [1, 2]})
            self.assertEqual(db.llm_cache_get("k1"), {"a": [1, 2]})
            # second put with the same key is ignored, not overwritten
            db.llm_cache_put("k1", "anthropic", "m", "medium", {"a": "other"})
            self.assertEqual(db.llm_cache_get("k1"), {"a": [1, 2]})

    def test_legacy_import(self):
        with temp_db() as db:
            self.assertEqual(db.llm_cache_import({"k1": {"x": 1},
                                                  "k2": {"y": 2}}), 2)
            self.assertEqual(db.llm_cache_import({"k2": {"y": 2},
                                                  "k3": {"z": 3}}), 1)
            self.assertEqual(db.llm_cache_get("k2"), {"y": 2})


class TestUsageHistory(unittest.TestCase):
    def test_append_and_load(self):
        rec = {"timestamp": "2026-07-13T10:00:00", "date": "2026-07-13",
               "product_line": "test", "provider": "anthropic",
               "extract_model": "m1", "group_model": "m2",
               "reviews": 10, "phrases": 20, "groups": 3,
               "calls": 5, "cache_hits": 2,
               "input_tokens": 100, "output_tokens": 50,
               "cache_read_tokens": 0, "cache_write_tokens": 0,
               "cost_usd": 0.5, "cost_known": True}
        with temp_db() as db:
            db.append_usage(rec)
            rows = db.load_usage()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], rec)


if __name__ == "__main__":
    unittest.main()
