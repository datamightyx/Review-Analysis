"""The concurrent (asyncio) pipeline passes must produce IDENTICAL results
regardless of task scheduling: gathered results are applied in submission
order, and group_phrases renumbers the ids minted by concurrent category
tasks to the ones a sequential run would assign."""
import asyncio
import re
import unittest

from pipeline.extract import extract_phrases
from pipeline.grouping import group_phrases, reassign_phrases
from pipeline.models import ExtractedPhrase, Review, Taxonomy

_PHRASE_RE = re.compile(r'\[(\d+)\] "([^"]+)"')


class FakeLLM:
    """Stands in for LLM: canned prompt-derived answers plus a seeded
    pseudo-random delay, so gather() completion order differs between
    seeds. Determinism holds iff outputs are identical across seeds."""

    def __init__(self, seed: int):
        self.seed = seed
        self.calls = 0

    async def json_call_async(self, system: str, user: str, schema: dict,
                              max_tokens: int = 16000) -> dict:
        self.calls += 1
        await asyncio.sleep((hash((self.seed, user)) % 7) * 0.004)
        return self.answer(system, user)


class FakeExtractor(FakeLLM):
    def answer(self, system: str, user: str) -> dict:
        out = []
        for m in re.finditer(r"\[(\d+)\] rating=.*? \| title: (.+)", user):
            out.append({"review_index": int(m.group(1)),
                        "phrases": [{"quote": m.group(2),
                                     "categories": ["positive"]}]})
        return {"reviews": out}


class FakeGrouper(FakeLLM):
    """Each phrase gets a group named by its first word (so phrases sharing
    a first word land in one group, exercising the reuse-by-name path)."""

    def answer(self, system: str, user: str) -> dict:
        return {"assignments": [
            {"phrase_index": int(i), "new_group_name": text.split()[0]}
            for i, text in _PHRASE_RE.findall(user)]}


class FakeSilent(FakeLLM):
    """Judge that skips every phrase — everything takes the fallback path."""

    def answer(self, system: str, user: str) -> dict:
        return {"assignments": []}


def _phrases() -> list[ExtractedPhrase]:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    out = []
    for i, w in enumerate(words):
        cat = "positive" if i % 2 == 0 else "negative"
        out.append(ExtractedPhrase(quote=f"{w} works {i}", category=cat,
                                   product="P", review_id=f"P:{i}"))
    return out


def _snapshot(tax: Taxonomy):
    return ([(g.id, g.category, g.name) for g in tax.groups.values()],
            [(c.id, c.group_id, c.text, dict(c.votes))
             for c in tax.canonicals.values()],
            tax._next_id)


class TestExtractDeterminism(unittest.TestCase):
    def test_phrase_order_is_batch_order(self):
        reviews = [Review(review_id=f"P:{i}", product="P", author="",
                          rating=5.0, title=f"title number {i}", date="",
                          variant="", body="irrelevant")
                   for i in range(10)]
        runs = []
        for seed in (0, 1, 2):
            phrases = extract_phrases(reviews, FakeExtractor(seed),
                                      batch_size=3)
            runs.append([(p.quote, p.review_id) for p in phrases])
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], runs[2])
        self.assertEqual([q for q, _ in runs[0]],
                         [f"title number {i}" for i in range(10)])


class TestGroupingDeterminism(unittest.TestCase):
    def test_ids_match_a_sequential_run(self):
        snaps = []
        for seed in (0, 1, 2):
            tax = group_phrases(_phrases(), Taxonomy(), FakeGrouper(seed),
                                batch_size=2, double_check=False)
            snaps.append(_snapshot(tax))
        self.assertEqual(snaps[0], snaps[1])
        self.assertEqual(snaps[0], snaps[2])
        # exactly what a sequential category-by-category run assigns:
        # positive (first phrase's category) first, ids alternating
        # group/canonical in creation order
        self.assertEqual(list(snaps[0][0][i][0] for i in range(6)),
                         ["g1", "g3", "g5", "g7", "g9", "g11"])
        self.assertEqual([c[0] for c in snaps[0][1]],
                         ["c2", "c4", "c6", "c8", "c10", "c12"])
        self.assertEqual(snaps[0][2], 13)
        by_id = {g[0]: g for g in snaps[0][0]}
        self.assertEqual(by_id["g1"][1:], ("positive", "alpha"))
        self.assertEqual(by_id["g7"][1:], ("negative", "bravo"))

    def test_log_and_audit_order_is_deterministic(self):
        runs = []
        for seed in (0, 1):
            log, audit = [], []
            group_phrases(_phrases(), Taxonomy(), FakeGrouper(seed),
                          batch_size=2, double_check=False,
                          log=log, audit=audit)
            runs.append((log, audit))
        self.assertEqual(runs[0], runs[1])
        # log ids were remapped to the renumbered canonicals
        self.assertEqual([e["canonical_id"] for e in runs[0][0]],
                         ["c2", "c4", "c6", "c8", "c10", "c12"])


class TestReassignDeterminism(unittest.TestCase):
    def test_fallback_rows_created_in_job_order(self):
        base = group_phrases(_phrases(), Taxonomy(), FakeGrouper(0),
                             batch_size=2, double_check=False)
        # reworded corpus: not an exact/gate match for any existing row, and
        # the judge skips everything -> every unique is fallback-placed
        reworded = [ExtractedPhrase(quote=p.quote.replace("works", "performs"),
                                    category=p.category, product=p.product,
                                    review_id=p.review_id)
                    for p in _phrases()]
        snaps = []
        for seed in (0, 1):
            import copy
            tax = copy.deepcopy(base)
            reassign_phrases(reworded, tax, FakeSilent(seed), batch_size=2)
            snaps.append(_snapshot(tax))
        self.assertEqual(snaps[0], snaps[1])
        # old rows lost all votes and were pruned; the fallback rows remain,
        # created in job order — category-major, like a sequential pass
        texts = [c[2] for c in snaps[0][1]]
        self.assertEqual(texts, [f"{w} performs {i}" for w, i in
                                 [("alpha", 0), ("charlie", 2), ("echo", 4),
                                  ("bravo", 1), ("delta", 3), ("foxtrot", 5)]])


if __name__ == "__main__":
    unittest.main()
