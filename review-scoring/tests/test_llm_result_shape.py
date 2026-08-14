"""A model that answers with a bare JSON array instead of the schema's
object must not crash the pass with `'list' object has no attribute 'get'`."""
import unittest
from pipeline.llm import _coerce_result
from pipeline.grouping import (GROUPING_SCHEMA, CONSOLIDATE_SCHEMA,
                               ROWMERGE_SCHEMA, VERIFY_SCHEMA)
from pipeline.extract import REPAIR_SCHEMA, build_extract_schema


class TestCoerceResult(unittest.TestCase):
    def test_object_passes_through_unchanged(self):
        got = {"assignments": [{"phrase_index": 0}]}
        self.assertIs(_coerce_result(got, GROUPING_SCHEMA), got)

    def test_bare_array_wrapped_into_sole_array_property(self):
        rows = [{"phrase_index": 1}, {"phrase_index": 0}]
        self.assertEqual(_coerce_result(rows, GROUPING_SCHEMA),
                         {"assignments": rows})

    def test_bare_array_wrapped_for_every_single_array_schema(self):
        for schema, key in ((VERIFY_SCHEMA, "decisions"),
                            (ROWMERGE_SCHEMA, "merges"),
                            (REPAIR_SCHEMA, "substrings"),
                            (build_extract_schema(), "reviews")):
            self.assertEqual(_coerce_result([], schema), {key: []}, key)

    def test_ambiguous_schema_raises_instead_of_guessing(self):
        # CONSOLIDATE_SCHEMA has two arrays (merges, moves) — a bare array
        # could be either, so refuse rather than silently drop half the work
        with self.assertRaises(RuntimeError):
            _coerce_result([{"keep_group_id": "g1"}], CONSOLIDATE_SCHEMA)

    def test_scalar_raises_with_payload_visible(self):
        with self.assertRaises(RuntimeError) as ctx:
            _coerce_result("no assignments found", GROUPING_SCHEMA)
        self.assertIn("no assignments found", str(ctx.exception))
        self.assertIn("assignments", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
