"""domain.yaml judge_examples: domain-supplied placement few-shots must reach
every judge prompt (grouping / consolidate / reassign), a custom domain must
never inherit another domain's examples, and --init-domain must export them
for editing."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import domain as domain_mod
from pipeline.grouping import _judge_examples_block, build_grouping_system


MINIMAL = {
    "categories": [{"id": "pro", "key": "quote"}],
    "sheets": [{"title": "Pro", "layout": "two_level", "categories": ["pro"],
                "headers": ["A"], "widths": [10.0]}],
}


class DomainExamplesTest(unittest.TestCase):
    def tearDown(self):
        domain_mod.set_active_domain(None)  # restore the default profile

    def test_default_domain_has_examples_in_grouping_prompt(self):
        dom = domain_mod.default_domain()
        self.assertTrue(dom.judge_examples)
        prompt = build_grouping_system(dom)
        self.assertIn("PLACEMENT EXAMPLES", prompt)
        self.assertIn(dom.judge_examples[0], prompt)

    def test_custom_domain_without_examples_has_no_block(self):
        dom = domain_mod.domain_from_dict(MINIMAL)
        self.assertEqual(dom.judge_examples, [])
        self.assertEqual(_judge_examples_block(dom), "")
        self.assertNotIn("PLACEMENT EXAMPLES", build_grouping_system(dom))

    def test_custom_domain_examples_parsed_and_rendered(self):
        data = dict(MINIMAL, judge_examples=["  my example -> group X  ", ""])
        dom = domain_mod.domain_from_dict(data)
        self.assertEqual(dom.judge_examples, ["my example -> group X"])
        self.assertIn("my example -> group X", _judge_examples_block(dom))

    def test_active_domain_feeds_the_block(self):
        domain_mod.set_active_domain(domain_mod.domain_from_dict(
            dict(MINIMAL, judge_examples=["custom reading"])))
        self.assertIn("custom reading", _judge_examples_block())
        self.assertIn("custom reading", build_grouping_system())

    def test_init_domain_yaml_round_trips_examples(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "domain.yaml"
            domain_mod.write_default_yaml(p)
            dom = domain_mod.load_domain(Path(td))
        self.assertEqual(dom.judge_examples,
                         domain_mod.default_domain().judge_examples)


if __name__ == "__main__":
    unittest.main()
