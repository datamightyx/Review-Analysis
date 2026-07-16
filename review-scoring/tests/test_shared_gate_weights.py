"""Cross-product gate feedback: all three layers are GLOBAL — rule weights
(layer 3) sum every product's ✓/✗, and exact/near precedents (layers 1-2)
transfer across products via aggregate_all_labels. Locks the design of
pipeline/precedents.aggregate_all_labels / aggregate_rule_weights /
GatePrecedents(shared_stats)."""
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.precedents import (GatePrecedents, aggregate_all_labels,
                                 aggregate_rule_weights, rebuild_shared_weights,
                                 SHARED_WEIGHTS_FILENAME)

REASON = "qualifier sets differ"


def _label(phrase, into, label, reason=REASON, category="positive"):
    return {"category": category, "phrase": phrase, "into": into,
            "reason": reason, "label": label}


def _write_product(root, name, records):
    d = root / name
    d.mkdir()
    labels = {f"{name}|{i}": rec for i, rec in enumerate(records)}
    (d / "gate_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    return d


class TestSharedWeights(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_weights_sum_across_products(self):
        # neither product alone reaches min_labels=3 for the rule, but the two
        # together are 3x "merge" (veto wrong) -> the rule softens globally
        _write_product(self.root, "prod_a", [
            _label("a1", "b1", "merge"),
            _label("a2", "b2", "merge"),
        ])
        _write_product(self.root, "prod_b", [
            _label("c1", "d1", "merge"),
        ])
        stats = aggregate_rule_weights(self.root)
        self.assertEqual(stats[REASON], [0, 3])

        # prod_a on its own would have only 2 merge labels -> not soft
        local = GatePrecedents({}, {"min_labels": 3, "soft_threshold": 0.7})
        self.assertFalse(local.rule_softness(REASON)[2])

        # with the shared tally installed, the rule is soft for prod_a too
        shared = GatePrecedents({}, {"min_labels": 3, "soft_threshold": 0.7},
                                stats)
        keep_n, merge_n, soft = shared.rule_softness(REASON)
        self.assertEqual((keep_n, merge_n), (0, 3))
        self.assertTrue(soft)
        # a soft rule lifts the veto (blocked=False)
        blocked, _ = shared.veto_verdict("positive", "x", "y", REASON)
        self.assertFalse(blocked)

    def test_exact_precedent_transfers_across_products(self):
        # prod_b labels the pair ("works great","appears it will work well")
        # as "keep" (veto correct). A run over prod_a must inherit that exact
        # verdict for the same pair, even though prod_a never labeled it.
        _write_product(self.root, "prod_a", [_label("x", "y", "merge")])
        _write_product(self.root, "prod_b", [
            _label("works great", "appears it will work well", "keep"),
        ])
        gp = GatePrecedents(aggregate_all_labels(self.root), None)
        blocked, basis = gp.veto_verdict(
            "positive", "works great", "appears it will work well", REASON)
        self.assertTrue(blocked)          # keep -> veto stands
        self.assertIn("точний прецедент", basis)

    def test_near_precedent_transfers_across_products(self):
        # prod_b labeled a pair "merge"; prod_a meets a lexically near pair
        # (same rule) and must inherit the merge verdict (veto lifted).
        _write_product(self.root, "prod_b", [
            _label("stops the bleeding fast", "stop bleeding fast", "merge",
                   reason=REASON),
        ])
        gp = GatePrecedents(aggregate_all_labels(self.root), None)
        blocked, basis = gp.veto_verdict(
            "positive", "stops the bleeding fast", "stop bleeding fast", REASON)
        self.assertFalse(blocked)  # exact hit here (same wording) -> merge
        # a lexically CLOSE but non-identical pair also inherits via layer 2
        blocked2, basis2 = gp.veto_verdict(
            "positive", "stops the bleeding fast!", "stop the bleeding fast",
            REASON)
        self.assertFalse(blocked2)
        self.assertTrue(basis2)

    def test_different_category_does_not_transfer(self):
        # category is still a guard: a "positive" precedent must not resolve a
        # "negative" pair.
        _write_product(self.root, "prod_b", [
            _label("works great", "appears it will work well", "keep",
                   category="positive"),
        ])
        gp = GatePrecedents(aggregate_all_labels(self.root), None)
        blocked, basis = gp.veto_verdict(
            "negative", "works great", "appears it will work well", REASON)
        self.assertTrue(blocked)   # default block, no evidence
        self.assertEqual(basis, "")

    def test_rebuild_writes_shared_file(self):
        _write_product(self.root, "prod_a", [_label("a1", "b1", "keep")])
        out = rebuild_shared_weights(self.root)
        self.assertEqual(out[REASON], [1, 0])
        f = self.root / SHARED_WEIGHTS_FILENAME
        self.assertTrue(f.exists())
        self.assertEqual(json.loads(f.read_text(encoding="utf-8"))[REASON],
                         [1, 0])

    def test_rebuild_removes_file_when_empty(self):
        f = self.root / SHARED_WEIGHTS_FILENAME
        f.write_text("{}", encoding="utf-8")
        rebuild_shared_weights(self.root)  # no product folders -> empty
        self.assertFalse(f.exists())


if __name__ == "__main__":
    unittest.main()
