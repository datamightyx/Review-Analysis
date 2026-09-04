"""Two-level grouping with explicit merge thresholds (the heart of the SOP).

Level 2 (canonical phrase): near-identical wordings are merged and votes
summed ("stop bleed quick" == "It stops the bleeding quickly"), while
meaningfully different qualifiers stay separate ("quickly" != "instantly").

Level 1 (group / USP statement): canonical phrases that express the same
theme are clustered; the group is named by the most common customer wording.

Deterministic pre-pass merges exact normalized duplicates for free; the LLM
judges everything else incrementally against the current taxonomy, so a new
competitor's phrases fill the existing structure instead of building a new one.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

from .llm import LLM, run_async
from .models import ExtractedPhrase, Group, Taxonomy
from .similarity import (auto_merge_target, fold_label, merge_blocked,
                         normalize, polarity_conflict, similarity,
                         top_candidates)
from . import embeddings
from . import precedents
from . import domain as _domain

# The merge rules below are product-agnostic: they turn on grammar, praise
# strength, negation and qualifier words — never on any product's vocabulary.
# Illustrations use only generic evaluative words ("works well" vs "works
# great", "quickly" vs "instantly") that mean the same in any category. The
# per-category meaning of a GROUP, and any usage buckets, are appended from
# the active domain profile by build_grouping_system().
_GROUPING_BASE = """\
You maintain a two-level taxonomy of customer review phrases for a
"review scoring" workbook.

Level 1 — GROUP: one broad theme (one selling point / one recurring
complaint / one usage type / one requested change) worth a single bullet
point.
Level 2 — CANONICAL PHRASE: one row inside a group. Near-identical customer
wordings are merged into one canonical and their votes are summed.

You receive the CURRENT TAXONOMY and a list of NEW PHRASES. For each new
phrase decide exactly one of:
 a) merge into an existing canonical (give canonical_id);
 b) new canonical inside an existing group (give group_id);
 c) new canonical inside a new group (give new_group_name).

MERGE INTO AN EXISTING CANONICAL only when it is the same message with
trivial differences:
 - typos, shorthand, word order, inflection, articles: a terse phrasing and
   a full-sentence phrasing of the SAME claim are one canonical.
 - praise wording of the SAME tier. Plain tier: good / well, bare
   unqualified effectiveness, and generic effectiveness idioms — "works
   good" == "Works well" == "they work" == "it really works" == "does the
   trick/job" == "gets the job done" == "the right tool for the job" -> ONE
   canonical (named by the most common wording, e.g. "Works well"). Strong
   tier: great / excellent / outstanding / amazing / wonderful / fantastic /
   awesome / perfect(ly) / flawless(ly) — "Works great" == "Work perfectly"
   == "These work excellent" -> ONE canonical. The tiers NEVER merge with
   each other: "Works well" vs "Works great" -> SEPARATE canonicals (in the
   same group).
 - a generic subject, beneficiary or future marker never blocks a praise
   merge — the message is the praise, not who or what it was for: "It worked
   fantastic", "they're gonna work great", "This worked great for her!!"
   are ALL the same message as "Works great" -> MERGE. Dropped-letter typos
   count as the full word: "works amazin" = amazing, "work perfectl" =
   perfectly.
 - a degree-intensifier swapped in front of the IDENTICAL core word: very /
   really / so / quite / extremely / pretty / surprisingly are interchangeable
   modifiers, not distinct content: "Works very well" == "These work really
   well!" == "This works so well" -> MERGE (the core word "well" is
   unchanged, only the intensifier differs). This applies ONLY when the word
   being intensified is the same word — swapping the core word itself
   (well/great/perfectly/fast/nicely) does NOT merge.
 - NEGATED complaints that state the same failure: negation form and degree
   words never split rows — "They just don't do it" == "Didn't do it very
   well" == "have no effect" == "It's not doing it" -> ONE canonical.
   Synonymous verbs for the SAME failure of the SAME aspect are one message.
   BUT a different object/aspect or an added timeframe still separates
   ("only lasts an hour or two" vs a generic failure) -> SEPARATE canonicals
   in the same group.
 - a short paraphrase carrying the SAME single claim, even with different
   words -> MERGE. This applies to claim-sized phrases only — a multi-detail
   descriptive sentence never dissolves into another wording (see DO NOT
   rules below).
 - a plain translation of an existing canonical into another language, when
   it names the same generic thing with no added nuance -> MERGE into that
   canonical. Judge this by MEANING, not spelling — translations share no
   lexical overlap with the English canonical, so the [similar: ...] hints
   will usually be empty for these; read the full CURRENT TAXONOMY list
   instead. Always keep the EXISTING canonical's text unchanged — never
   rewrite it into the new language, and never add a second row that is just
   its translation. A foreign-language phrase that adds real nuance no
   existing canonical covers still gets its own canonical, in its original
   wording.
DO NOT merge (keep separate canonicals, usually in the same group) when:
 - a meaningful qualifier differs: "quickly" vs "instantly/immediately" ->
   different adverb, SEPARATE; two different timeframes -> SEPARATE;
 - the CORE WORD itself differs and it is not a same-tier praise swap:
   "Works great" vs "Works well" vs "worked ok" -> SEPARATE ("well" is plain
   praise, "great"/"perfectly" are strong praise — different tiers; fast /
   ok / quickly belong to no tier and never swap for anything — only the
   intensifier itself is interchangeable, per the MERGE rules above);
 - the QUALIFIER is a DIFFERENT WORD, even a synonym: "fast" vs "quickly"
   -> SEPARATE ("fast" is not an inflection of "quick"); "right away" vs
   "instantly" -> SEPARATE. Qualifiers are speed / time / degree words
   (fast, quickly, instantly, ok, overnight, for a week) — they carry the
   customer's exact claim strength and never swap. This rule is about
   qualifiers, NOT about the plain verb of the claim (synonymous failure /
   benefit verbs for the same aspect are one message);
 - different object, part, mechanism or context -> SEPARATE;
 - one phrase is short/generic and the other is a longer descriptive
   sentence: they stay SEPARATE canonicals in the same group — detailed
   wordings are valuable content material and must survive as their own rows;
 - both are LONG multi-detail descriptive sentences worded differently —
   each carries its own vivid details, SEPARATE rows. (Short claim-sized
   phrases with the same single message DO merge — see the MERGE rules
   above.) When in doubt, keep separate rows in the same group.

GROUPS ARE BROAD. A group is a whole theme (one point worth one bullet on a
product page), NOT a phrasing variant. All nuance lives at the canonical
level INSIDE the group: many differently-worded rows about the same benefit
belong to ONE group as separate canonicals.

NEVER create a group whose theme is a synonym or sub-case of an existing
group. If an existing group covers the same customer benefit — use it. A
typical product line ends up with roughly 10-25 groups per category, most
phrases landing in the big early groups. Create a new group ONLY for a
genuinely different benefit / problem / usage.

DOMINANT ANGLE. When a quote touches two themes, place it by the angle that
carries its distinctive content, NOT by a feature it merely mentions in
passing (e.g. a phrase whose real point is KEEPING the product for an
occasion, or HOW it is applied, belongs to the keep-on-hand / ease-of-use
theme even if it also names the core benefit in passing). Do not split one
theme into several groups by occasion or by who it is for. Use dual
placement only when both themes genuinely carry weight.
A complaint blaming a CHANGE (recipe, formula, materials, packaging) for the
product failing at its core job belongs under the theme for THAT failure
(e.g. "doesn't work anymore"), not a standalone "they changed it" group —
unless enough reviews complain about the change itself as the point, not
just as the blamed cause.

NEW GROUP NAME: short (2-5 words), preferring the most frequent customer
wording.

DUAL PLACEMENT (two groups): if one phrase genuinely covers two distinct
themes ("durable and reusable", "good size and quality"), fill second_group_id
or second_group_new_name — the vote will be counted in both groups.

DUAL PLACEMENT (two rows of one group): if a quote names two benefits that
already exist as SEPARATE canonical rows in the SAME group — e.g. "Great for
an emergency or first aid kit" when the group has both a "...for emergencies"
row and a "...first aid kit" row — put it in the row that carries its primary
wording via canonical_id/group_id, and set second_canonical_id (or
second_canonical_text) to the OTHER existing row. The vote counts in both
rows, each keeping its own wording. Use this only when both named benefits
already have their own rows; do not invent a second row for a benefit merely
mentioned in passing (the DOMINANT ANGLE rule still applies).

ANSWER PROTOCOL (strict):
 - canonical_id and group_id must be ids copied VERBATIM from the CURRENT
   TAXONOMY above (they look like c12, g3). Never invent or abbreviate ids.
 - Phrases within one batch often belong together — cluster them! To place
   a phrase into a group you are creating for an EARLIER phrase of THIS
   batch, set same_group_as_phrase to that phrase's index (leave group_id
   empty). Repeating the exact same new_group_name string works too.
 - To merge a phrase into the canonical created for an earlier phrase of
   THIS batch, set merge_with_phrase to that phrase's index.
 - Some phrases carry a [similar: ...] hint — existing canonicals that are
   lexically close. Check those entries (and their groups) FIRST, but treat
   them as hints only: lexical closeness does not override the merge rules
   ("quickly" vs "instantly" stay separate canonicals).

Be consistent: identical inputs must produce identical decisions.
"""

# Universal placement rules. Product-agnostic like _GROUPING_BASE (they turn
# on the strength and direction of a claim, never on any product's
# vocabulary), but shown to EVERY judge that places or re-places a phrase:
# grouping, consolidation, reassignment and splitting. A rule only one pass
# sees is a rule the next pass undoes — the same reason _judge_examples_block
# is appended everywhere.
#
# Both rules are defect classes measured against a human-scored workbook for
# the same product line: an illness filed under the group naming the extreme
# outcome, a mechanism invented for a bare "it made him sick", and a
# complaint that the product clumps TOO hard filed under the "does not clump"
# theme. Domain vocabulary for these readings belongs in domain.yaml
# judge_examples; the grammar of the mistake is universal and lives here.
_PLACEMENT_RULES = """\

CLAIM FIDELITY — never round a claim up. Place a phrase by what it literally
says, never by a stronger, more specific or more severe version of it.
 - Do not escalate an outcome or a degree: a fault, an illness or a hazard is
   not the worst case of its kind unless the customer states that worst case.
   A phrase reporting that something went wrong belongs with the general
   problem, NOT with the group named after the extreme outcome.
 - Do not invent a mechanism, cause, affected part or component the wording
   does not name. That something went wrong is not evidence of WHICH thing
   went wrong.
 - If only a severe or highly specific group exists and the phrase is
   general, create the missing general group instead of forcing the phrase
   into the severe one.

OPPOSITE CLAIMS NEVER SHARE A GROUP. Two phrases asserting opposite
directions of the SAME attribute — too much of it vs too little, present vs
absent, works vs fails — are two different themes even when they use the same
nouns. Judge the DIRECTION of the claim, not the shared vocabulary. A group
whose name states one direction must never hold a row stating the other; when
the opposite theme has no group yet, create it.
"""


def _category_notes(dom: _domain.Domain | None = None) -> str:
    """Per-category guidance appended to the judge prompts, generated from
    the active domain profile: what a GROUP means in each category, and (for
    relation categories with a coarse bucket) the usage_category instruction.
    Keeps all product structure in data, not in the prompt text."""
    dom = dom or _domain.active()
    lines = ["CATEGORY NOTES (the meaning of a GROUP per category):"]
    for c in dom.categories:
        hint = c.group_hint or "a distinct theme."
        lines.append(f" - {c.id}: the group is {hint}")
        if c.subbucket:
            lines.append(
                f"   For {c.id} also set usage_category — a coarse bucket "
                f"grouping related {c.id} types together (reuse buckets "
                f"already present in the taxonomy when they fit).")
    return "\n".join(lines)


def _judge_examples_block(dom: _domain.Domain | None = None) -> str:
    """Domain-supplied placement few-shots (domain.yaml judge_examples),
    appended to every judge prompt. The grouping, reassignment AND
    consolidation judges must all see the same examples — a pass without
    them undoes the placements the others got right (measured on the
    reference workbook)."""
    dom = dom or _domain.active()
    if not dom.judge_examples:
        return ""
    lines = ["PLACEMENT EXAMPLES from this product domain (human-approved "
             "readings — apply the same logic to similar phrases):"]
    lines += [f" - {e}" for e in dom.judge_examples]
    return "\n".join(lines) + "\n"


def build_grouping_system(dom: _domain.Domain | None = None) -> str:
    return (_GROUPING_BASE + _PLACEMENT_RULES + "\n" + _category_notes(dom)
            + "\n" + _judge_examples_block(dom))

GROUPING_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phrase_index": {"type": "integer"},
                    "canonical_id": {"type": "string"},
                    "group_id": {"type": "string"},
                    "new_group_name": {"type": "string"},
                    "same_group_as_phrase": {"type": "integer"},
                    "merge_with_phrase": {"type": "integer"},
                    "second_group_id": {"type": "string"},
                    "second_group_new_name": {"type": "string"},
                    "second_canonical_id": {"type": "string"},
                    "second_canonical_text": {"type": "string"},
                    "usage_category": {"type": "string"},
                },
                "required": ["phrase_index"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}


def polarity_conflicts(tax: Taxonomy, category: str) -> list[tuple[str, str, str]]:
    """(group_id, row_a_text, row_b_text) for every pair of rows inside one
    group of `category` that make OPPOSITE claims about a tracked attribute
    (config.yaml: attribute_families) — e.g. "unscented" and "smells
    amazing" sitting in the same group. No-op (empty list) when no families
    are configured. Used by split_groups_async to force-review a group
    regardless of its size, and by pipeline/quality.py to warn a human."""
    out: list[tuple[str, str, str]] = []
    for g in tax.groups_for(category):
        cans = g.canonicals(tax)
        for i, a in enumerate(cans):
            for b in cans[i + 1:]:
                if polarity_conflict(a.text, b.text):
                    out.append((g.id, a.text, b.text))
    return out


def _unique_phrases(phrases: list[ExtractedPhrase]):
    """Deterministic pre-merge of exact normalized duplicates.

    Returns list of dicts: {text, category, counts: {product: n},
    raws: {product: [raw quotes]}, relation, gist} sorted by count desc.
    """
    buckets: dict[tuple, dict] = {}
    dom = _domain.active()
    for p in phrases:
        key_text = p.relation if dom.is_relation(p.category) and p.relation else p.quote
        key = (p.category, normalize(key_text))
        b = buckets.setdefault(key, {
            "text": key_text.strip(),
            "category": p.category,
            "counts": defaultdict(int),
            "raws": defaultdict(list),
            "review_ids": defaultdict(list),
            # (review_id, quote) pairs exactly as extracted — the only place
            # the true per-review pairing survives; raws/review_ids above
            # are each deduped independently and lose it (see Canonical.
            # quote_sources docstring in models.py)
            "pairs": defaultdict(list),
            "relation": p.relation,
            "gist": p.gist,
            "quote_en": "",
        })
        b["counts"][p.product] += 1
        if not b["quote_en"] and p.quote_en:
            b["quote_en"] = p.quote_en.strip()
        if p.quote not in b["raws"][p.product]:
            b["raws"][p.product].append(p.quote)
        if p.review_id and p.review_id not in b["review_ids"][p.product]:
            b["review_ids"][p.product].append(p.review_id)
        if p.review_id and p.quote:
            b["pairs"][p.product].append((p.review_id, p.quote))
    result = list(buckets.values())
    result.sort(key=lambda b: -sum(b["counts"].values()))
    return result


def _taxonomy_context(tax: Taxonomy, category: str,
                      max_canonicals: int | None = 12) -> str:
    """Groups with their top canonicals. Long tails are truncated (when
    max_canonicals is an int) to keep the judge's attention on the
    structure during incremental grouping/reassign, where a
    [similar: ...] hint surfaces relevant tail entries per-phrase anyway.
    max_canonicals=None shows every row — required for consolidate_taxonomy
    and split_groups_async, which propose MOVING/SPLITTING rows: a row the
    prompt never shows can never be moved (measured: on a 26-group, ~470-row
    taxonomy, 9 groups exceeded the old max_canonicals=30 cutoff — exactly
    the groups later found to need splitting/moving, because consolidation
    was structurally blind to their tail rows)."""
    groups = tax.groups_for(category)
    if not groups:
        return "(empty — this is the first batch, create groups as needed)"
    lines = []
    for g in sorted(groups, key=lambda g: -g.total(tax)):
        head = f"GROUP {g.id}: {g.name}"
        if g.usage_category:
            head += f"  [usage_category: {g.usage_category}]"
        lines.append(head)
        cans = sorted(g.canonicals(tax), key=lambda c: -c.total)
        shown = cans if max_canonicals is None else cans[:max_canonicals]
        for c in shown:
            lines.append(f"  {c.id}: {c.text}  ({c.total})")
        if max_canonicals is not None and len(cans) > max_canonicals:
            lines.append(f"  … and {len(cans) - max_canonicals} more rows")
    return "\n".join(lines)


def _candidate_hint(tax: Taxonomy, category: str, text: str) -> str:
    """Shortlist of similar existing canonicals for one phrase: lexical
    top-5 plus (when the optional embedding model is available) up to 3
    semantic neighbours — those catch paraphrases that share no words
    ("right tool for the job" ~ "Work well")."""
    items = [(c.id, c.text, c.group_id) for g in tax.groups_for(category)
             for c in g.canonicals(tax)]
    return _hint_from_items(items, text)


def _hint_from_items(items: list[tuple[str, str, str]], text: str) -> str:
    """_candidate_hint over a frozen (id, text, group_id) snapshot — the
    reassign pass snapshots its candidate pool once so all batch prompts can
    be built (and their calls gathered) up front."""
    pairs = [(cid, ctext) for cid, ctext, _ in items]
    cands = top_candidates(text, pairs, k=5)
    seen = {cid for cid, _, _ in cands}
    cands += [c for c in embeddings.top_semantic(text, pairs, k=3)
              if c[0] not in seen]
    if not cands:
        return ""
    gid_of = {cid: gid for cid, _, gid in items}
    parts = []
    for cid, ctext, _ in cands:
        parts.append(f'{cid} "{ctext}" in {gid_of[cid]}')
    return "  [similar: " + "; ".join(parts) + "]"


def _exact_map(tax: Taxonomy, category: str) -> dict[str, list]:
    """normalized canonical text -> all canonicals carrying it (dual-placement
    twins included) within one category."""
    by_text: dict[str, list] = defaultdict(list)
    for g in tax.groups_for(category):
        for c in g.canonicals(tax):
            by_text[normalize(c.text)].append(c)
    return by_text


def _add_unique(canon, b: dict) -> None:
    for product, count in b["counts"].items():
        canon.add(product, count, "; ".join(b["raws"][product][:3]),
                  b["review_ids"].get(product), b.get("pairs", {}).get(product))


def _group_named(tax: Taxonomy, category: str, name: str):
    """The group of `category` already carrying `name`, or None.

    Every path that creates a group BY NAME must consult this first — two
    groups with one name in one category render as two separate blocks under
    the same label in the workbook, which is never what was meant. Compared
    with fold_label, not normalize: a group name is a THEME label, and two
    themes differing only in grammatical number ("Chinchilla" /
    "Chinchillas", "Bigger amount" / "Bigger amounts") are one theme. The
    rows inside keep their own exact wording either way."""
    n = fold_label(name)
    return next((g for g in tax.groups_for(category)
                 if fold_label(g.name) == n), None)


# A judge-proposed group name longer than this is treated as a pasted quote
# rather than a theme label. The prompt asks for 2-5 words; the cap leaves
# slack for hyphenation and short connectives.
_MAX_GROUP_NAME_WORDS = 6


def _group_name_for(category: str, b: dict, proposed: str) -> str:
    """Name for a group being created for phrase bucket `b`.

    For a category named from the extracted `gist` label (domain.yaml
    name_from: gist — the improvement wishes), the SHORT label wins over a
    long judge-proposed name. Models routinely echo the raw customer sentence
    back as new_group_name, and because `proposed` used to win outright every
    wish became its own group named by a verbatim quote — leaving the
    consolidation pass nothing to match themes on (measured on the Hamster
    Sand workbook 2026-08-17: 12 improvement groups, 8 named by a full quote,
    two pairs of them the same wish). A proposed name that is already short is
    still preferred: it is the pass that sees the whole taxonomy, so it can
    reuse a theme label that exists. Falls back to the phrase text when
    neither is usable; categories not named from a gist are unaffected."""
    gist = (b.get("gist") or "").strip()
    if gist and _domain.active().name_from(category) == "gist":
        if proposed and len(proposed.split()) <= _MAX_GROUP_NAME_WORDS:
            return proposed
        return gist
    return proposed or b["text"]


def _deterministic_prepass(tax: Taxonomy, category: str, uniques: list[dict],
                           audit: list | None = None) -> list[dict]:
    """Free placements before any judge call. Exact normalized matches go
    straight to their rows (all dual-placement twins get the vote); then
    wordings the gate certifies as the same message ("They work good" ->
    "Works well", "The incision worked fantastic" -> "Works great")
    auto-merge into their row. Returns the phrases the judge still has to
    see. usage/who_recommended rows are relations, not verbatim wordings,
    so only the exact pass applies there."""
    by_text = _exact_map(tax, category)
    gate_applies = _domain.active().gated(category)
    use_gist = _domain.active().name_from(category) == "gist"
    items = [(c.id, c.text) for g in tax.groups_for(category)
             for c in g.canonicals(tax)] if gate_applies else []
    pending = []
    for b in uniques:
        twins = by_text.get(normalize(b["text"]))
        if twins:
            for c in twins:
                _add_unique(c, b)
            continue
        hit = auto_merge_target(b["text"], items) if gate_applies else None
        if hit is not None:
            target = tax.canonicals[hit[0]]
            for c in by_text.get(normalize(target.text), [target]):
                _add_unique(c, b)
            if audit is not None:
                g = tax.groups.get(target.group_id)
                audit.append({"type": "auto_merge", "category": category,
                              "phrase": b["text"], "into": target.text,
                              "group": g.name if g else "?"})
            continue
        # a wish whose gist names a theme that ALREADY exists goes straight
        # into that group — its own row, no judge call. This is what makes
        # `gist` the clustering key of a name_from: gist category instead of
        # a naming hint the judge is free to ignore; without it every wish
        # was judged on its sentence alone and became its own group
        # (Hamster Sand 2026-08-17: 12 improvement groups for 5 real themes).
        # The row keeps the customer's verbatim wording — only the GROUP is
        # decided by the gist.
        if use_gist and b.get("gist"):
            g = _group_named(tax, category, b["gist"])
            if g is not None:
                _add_unique(tax.new_canonical(b["text"], g.id), b)
                if audit is not None:
                    audit.append({"type": "gist_match", "category": category,
                                  "phrase": b["text"], "into": b["gist"],
                                  "group": g.name})
                continue
        pending.append(b)
    return pending


def consolidate_rows(tax: Taxonomy, audit: list | None = None) -> list[str]:
    """Deterministic row-level cleanup: existing canonicals that the merge
    gate certifies as the same message are merged into the highest-voted
    wording (which then names the row). Catches sibling rows the judge
    created instead of merging, and duplicates left in taxonomies built
    before the gate learned a rule. Dual-placement twins (same normalized
    text in two groups) are deliberate and never touched."""
    actions: list[str] = []
    dom = _domain.active()
    for category in dom.ids():
        if not dom.gated(category):
            continue
        cans = [c for g in tax.groups_for(category)
                for c in g.canonicals(tax)]
        cans.sort(key=lambda c: -c.total)
        # rows with an exact-text twin are dual placements: the same vote
        # deliberately counted in two groups — merging either copy would
        # double-count or destroy the second theme, so leave them alone
        text_count = defaultdict(int)
        for c in cans:
            text_count[normalize(c.text)] += 1
        gone: set[str] = set()
        for i, keep in enumerate(cans):
            if keep.id in gone or text_count[normalize(keep.text)] > 1:
                continue
            for other in cans[i + 1:]:
                if other.id in gone or text_count[normalize(other.text)] > 1:
                    continue
                if auto_merge_target(other.text,
                                     [(keep.id, keep.text)]) is None:
                    continue
                g_to = tax.groups.get(keep.group_id)
                actions.append(f"[{category}] рядок «{other.text}» злито в "
                               f"«{keep.text}»")
                if audit is not None:
                    audit.append({"type": "auto_merge", "category": category,
                                  "phrase": other.text, "into": keep.text,
                                  "group": g_to.name if g_to else "?"})
                _merge_canonical_into(tax, keep, other)
                gone.add(other.id)
    return actions


def group_phrases(phrases: list[ExtractedPhrase], tax: Taxonomy, llm: LLM,
                  batch_size: int = 25, progress=None,
                  log: list | None = None,
                  double_check: bool = True,
                  audit: list | None = None,
                  verify_llm: LLM | None = None,
                  checkpoint=None) -> Taxonomy:
    """`checkpoint`, if given, is called (no args) after every applied batch
    so a crash mid-pass loses at most one batch's worth of merges instead of
    the whole pass — typically `lambda: db.save_taxonomy(tax)`."""
    return run_async(group_phrases_async(
        phrases, tax, llm, batch_size=batch_size, progress=progress, log=log,
        double_check=double_check, audit=audit, verify_llm=verify_llm,
        checkpoint=checkpoint))


async def group_phrases_async(phrases: list[ExtractedPhrase], tax: Taxonomy,
                              llm: LLM, batch_size: int = 25, progress=None,
                              log: list | None = None,
                              double_check: bool = True,
                              audit: list | None = None,
                              verify_llm: LLM | None = None,
                              checkpoint=None) -> Taxonomy:
    """Categories never share groups, so each category runs as its own task
    and the tasks are gathered concurrently. WITHIN a category the batches
    stay sequential — every batch's prompt shows the taxonomy as mutated by
    the previous batch. Determinism: per-category log/audit entries are
    concatenated in category order, and _renumber_new() remaps the ids
    minted this pass to the ones a sequential category-by-category run
    would have assigned (the shared id counter is the only cross-category
    coupling)."""
    # the double-check is a narrow "is this new group a duplicate?" decision —
    # it may run on a cheaper/lower-effort LLM than the grouping judge
    verify_llm = verify_llm or llm
    by_cat: dict[str, list[ExtractedPhrase]] = defaultdict(list)
    for p in phrases:
        by_cat[p.category].append(p)

    # votes/quotes are rebuilt from the full current corpus on every run:
    # the taxonomy carries STRUCTURE (groups + row texts) across runs;
    # carrying votes too would double-count phrases on a rerun
    for c in tax.canonicals.values():
        c.votes, c.quotes, c.review_ids = {}, {}, {}

    preexisting = set(tax.groups) | set(tax.canonicals)
    id_start = tax._next_id
    cat_logs = {c: [] if log is not None else None for c in by_cat}
    cat_audits = {c: [] if audit is not None else None for c in by_cat}

    await asyncio.gather(*(
        _group_category(tax, category, cat_phrases, llm, verify_llm,
                        batch_size, progress, cat_logs[category],
                        double_check, cat_audits[category], checkpoint)
        for category, cat_phrases in by_cat.items()))

    # merge the per-task journals back in deterministic (category) order
    for category in by_cat:
        if audit is not None:
            audit.extend(cat_audits[category])
        if log is not None:
            log.extend(cat_logs[category])

    # sibling rows that are the same message per the gate collapse into one
    consolidate_rows(tax, audit)
    # rows that got no votes from the current corpus are stale
    prune_empty(tax)
    _renumber_new(tax, preexisting, id_start, list(by_cat), log)
    return tax


async def _group_category(tax: Taxonomy, category: str,
                          cat_phrases: list[ExtractedPhrase], llm: LLM,
                          verify_llm: LLM, batch_size: int, progress,
                          log: list | None, double_check: bool,
                          audit: list | None, checkpoint=None) -> None:
    uniques = _unique_phrases(cat_phrases)
    # deterministic pre-pass (exact matches + gate-certified merges) —
    # no judge call, identical inputs stay consistent and reruns cost
    # nothing for already-known wordings
    pending = _deterministic_prepass(tax, category, uniques, audit)
    done = len(uniques) - len(pending)
    if progress and done:
        progress(category, done, len(uniques))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        ctx = _taxonomy_context(tax, category)
        plines = []
        for j, b in enumerate(batch):
            total = sum(b["counts"].values())
            extra = ""
            if _domain.active().name_from(category) == "gist" and b["gist"]:
                extra = f"  [wish: {b['gist']}]"
            if _domain.active().is_relation(category):
                sample = next(iter(
                    q for qs in b["raws"].values() for q in qs), "")
                extra = f'  [quote: "{sample}"]'
            if b.get("quote_en"):
                extra += f'  [translation: "{b["quote_en"]}"]'
            # match on the English gloss when the phrase itself isn't
            # English — a foreign quote has no lexical/embedding overlap
            # with English canonicals, so [similar: ...] would otherwise
            # come back empty and the judge never sees the candidate at all
            extra += _candidate_hint(tax, category, b.get("quote_en") or b["text"])
            plines.append(f'[{j}] "{b["text"]}" (x{total}){extra}')
        user = (
            f"Category: {category}\n\n"
            f"CURRENT TAXONOMY:\n{ctx}\n\n"
            f"NEW PHRASES:\n" + "\n".join(plines)
        )
        result = await llm.json_call_async(build_grouping_system(), user,
                                           GROUPING_SCHEMA)

        # apply the whole batch in one synchronous block (no awaits), so
        # concurrent category tasks can never observe it half-applied.
        # process in phrase order so within-batch references
        # (same_group_as_phrase / merge_with_phrase) resolve
        groups_before = set(tax.groups)
        assignments = sorted(result.get("assignments", []),
                             key=lambda a: a.get("phrase_index", 0))
        batch_canon: dict[int, "object"] = {}
        for a in assignments:
            idx = a.get("phrase_index", -1)
            if not 0 <= idx < len(batch):
                continue
            b = batch[idx]
            canon = _apply_assignment(tax, category, b, a, batch_canon,
                                      audit=audit)
            batch_canon[idx] = canon
            if log is not None:
                log.append({"category": category, "text": b["text"],
                            "canonical_id": canon.id})
        # a phrase the judge skipped must not lose its votes — park it
        # next to its lexically closest row and flag it for review
        for j, b in enumerate(batch):
            if j not in batch_canon:
                _fallback_place(tax, category, b, audit)
        new_ids = [gid for gid in tax.groups if gid not in groups_before]

        if double_check:
            await _verify_new_groups(tax, category, new_ids, verify_llm)

        done += len(batch)
        if progress:
            progress(category, done, len(uniques))
        # durably persist structure after every applied batch — a crash
        # (OOM kill, container restart, ...) loses at most this one batch
        # instead of the whole pass; cheap relative to the LLM call above
        if checkpoint:
            checkpoint()


def _renumber_new(tax: Taxonomy, preexisting: set[str], start: int,
                  cat_order: list[str], log: list | None = None) -> None:
    """Concurrent category tasks draw from the taxonomy's single id counter,
    so the NUMBERS minted this pass depend on task scheduling. Remap them to
    the ids a sequential category-by-category run would assign — category
    order first, then creation order within the category (which numeric
    order preserves, since each task is internally sequential) — and rebuild
    the dicts (surviving pre-existing entries first, in their original
    order) so ids, dict order and therefore Excel output are identical for
    identical inputs regardless of concurrency."""
    cat_idx = {c: i for i, c in enumerate(cat_order)}

    def sort_key(obj_id: str, category: str) -> tuple[int, int]:
        return (cat_idx.get(category, len(cat_order)), int(obj_id[1:]))

    new_ids: list[tuple[tuple[int, int], str]] = []
    for gid, g in tax.groups.items():
        if gid not in preexisting:
            new_ids.append((sort_key(gid, g.category), gid))
    for cid, c in tax.canonicals.items():
        if cid not in preexisting:
            g = tax.groups.get(c.group_id)
            new_ids.append((sort_key(cid, g.category if g else ""), cid))
    if not new_ids:
        return
    new_ids.sort()
    mapping: dict[str, str] = {}
    n = start
    for _, old in new_ids:
        mapping[old] = f"{old[0]}{n}"   # keep the g/c prefix
        n += 1

    groups: dict[str, Group] = {}
    for gid, g in tax.groups.items():
        if gid in preexisting:
            groups[gid] = g
    canons: dict[str, Canonical] = {}
    for cid, c in tax.canonicals.items():
        if c.group_id in mapping:
            c.group_id = mapping[c.group_id]
        if cid in preexisting:
            canons[cid] = c
    for _, old in new_ids:
        new = mapping[old]
        if old in tax.groups:
            g = tax.groups[old]
            g.id = new
            groups[new] = g
        else:
            c = tax.canonicals[old]
            c.id = new
            canons[new] = c
    tax.groups = groups
    tax.canonicals = canons
    tax._next_id = n
    if log:
        for entry in log:
            cid = entry.get("canonical_id")
            if cid in mapping:
                entry["canonical_id"] = mapping[cid]


def _fallback_place(tax: Taxonomy, category: str, b: dict,
                    audit: list | None = None):
    """Last-resort placement (judge skipped the phrase or ids were invalid):
    own row in the group of the lexically closest existing row, or a fresh
    group when the category is empty. Flagged for human review."""
    items = [(c.id, c.text) for g in tax.groups_for(category)
             for c in g.canonicals(tax)]
    cands = top_candidates(b["text"], items, k=1, min_sim=0.0)
    if cands:
        group = tax.groups[tax.canonicals[cands[0][0]].group_id]
    else:
        name = _group_name_for(category, b, "")
        group = _group_named(tax, category, name) or tax.new_group(category,
                                                                   name)
    canon = tax.new_canonical(b["text"], group.id)
    _add_unique(canon, b)
    if audit is not None:
        audit.append({"type": "fallback", "category": category,
                      "phrase": b["text"], "group": group.name})
    return canon


def prune_empty(tax: Taxonomy) -> list[str]:
    """Drop canonicals with zero votes and then groups left without rows.
    Returns human-readable descriptions of what was removed."""
    actions: list[str] = []
    for c in list(tax.canonicals.values()):
        if c.total == 0:
            g = tax.groups.get(c.group_id)
            cat = g.category if g else "?"
            actions.append(f"[{cat}] рядок «{c.text}» лишився без голосів — "
                           f"видалено")
            del tax.canonicals[c.id]
    for g in list(tax.groups.values()):
        if not g.canonicals(tax):
            actions.append(f"[{g.category}] порожню групу «{g.name}» видалено")
            del tax.groups[g.id]
    return actions


VERIFY_SYSTEM = """\
You audit a two-level taxonomy of customer review phrases (GROUP = one broad
marketing theme worth one bullet point; rows inside carry the nuance).

You get the EXISTING GROUPS and the NEWLY CREATED groups from the latest
batch. For each new group decide:
 - keep: it is a genuinely different benefit / problem / usage theme;
 - merge_into: it is a synonym, rewording, or sub-case of an existing group
   (a narrower or reworded version of a benefit an existing group already
   covers). Give the target_group_id copied VERBATIM from the list.
Do NOT merge groups that represent different benefits even if they share
words (a group about EASE OF USE vs a group about STORAGE stay separate).
When unsure, keep.
"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "action": {"type": "string", "enum": ["keep", "merge_into"]},
                    "target_group_id": {"type": "string"},
                },
                "required": ["group_id", "action"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def _merge_canonical_into(tax: Taxonomy, keep, other) -> None:
    """Merge `other`'s votes/quotes into `keep`, then drop `other`. Never
    touches `keep.text` — the surviving phrase's wording is untouched."""
    for product, count in other.votes.items():
        keep.votes[product] = keep.votes.get(product, 0) + count
    for product, quotes in other.quotes.items():
        keep.quotes.setdefault(product, [])
        for q in quotes:
            if q not in keep.quotes[product]:
                keep.quotes[product].append(q)
    _merge_review_ids(keep, other)
    _merge_quote_sources(keep, other)
    _merge_translations(keep, other)
    # one review = one vote: a review present in BOTH canonicals was just
    # summed above but deduped in review_ids — re-derive the count from the
    # distinct ids so the shared review is not counted twice
    for product, ids in keep.review_ids.items():
        if ids:
            keep.votes[product] = len(ids)
    del tax.canonicals[other.id]


def _merge_review_ids(keep, other) -> None:
    for product, ids in other.review_ids.items():
        dst = keep.review_ids.setdefault(product, [])
        for rid in ids:
            if rid not in dst:
                dst.append(rid)


def _merge_quote_sources(keep, other) -> None:
    for product, smap in other.quote_sources.items():
        dst = keep.quote_sources.setdefault(product, {})
        for quote, rids in smap.items():
            lst = dst.setdefault(quote, [])
            for rid in rids:
                if rid not in lst:
                    lst.append(rid)


def _merge_translations(keep, other) -> None:
    for q, en in other.translations.items():
        keep.translations.setdefault(q, en)


def _dedupe_canonical_into(tax: Taxonomy, keep, other) -> None:
    """Absorb `other` into `keep` without double-counting. Two canonicals
    with the exact same normalized text inside one category can only exist
    because of dual placement (the same quote deliberately copied into a
    second group) — so `other` is the same underlying vote as `keep`, not an
    additional one. Only fill in products/quotes `keep` doesn't already
    carry; never add to a count `keep` already has."""
    for product, count in other.votes.items():
        if product not in keep.votes:
            keep.votes[product] = count
    for product, quotes in other.quotes.items():
        keep.quotes.setdefault(product, [])
        for q in quotes:
            if q not in keep.quotes[product]:
                keep.quotes[product].append(q)
    _merge_review_ids(keep, other)
    _merge_quote_sources(keep, other)
    _merge_translations(keep, other)
    del tax.canonicals[other.id]


def find_quote_key(canon, quote: str) -> tuple[str, str] | None:
    """(product, exact key) of `quote` inside `canon.quote_sources`, or None.
    `quote_sources` is the complete per-review pairing (grouping records every
    (review_id, quote) pair), unlike `quotes`, which keeps at most 3 joined
    samples for display."""
    want = normalize(quote)
    for product, smap in canon.quote_sources.items():
        for q in smap:
            if normalize(q) == want:
                return product, q
    return None


def _refresh_quote_sample(canon, product: str) -> None:
    """Rebuild the display sample `quotes[product]` from what is left in
    quote_sources — entries there are "; "-joined triples, so a raw quote can
    never simply be list-removed."""
    keys = list(canon.quote_sources.get(product, {}))
    if keys:
        canon.quotes[product] = ["; ".join(keys[:3])]
    else:
        canon.quotes.pop(product, None)


def move_quote_between(tax: Taxonomy, src, dst, quote: str) -> int:
    """Detach one RAW review quote from `src` and hand it — together with the
    reviews that actually said it — to `dst`.

    Votes follow the reviews: a review stays in `src` only while some other
    surviving quote of that row still carries it. Returns how many reviews
    moved. `src` may be left with zero votes; the caller prunes."""
    want = normalize(quote)
    moved = 0
    for product, smap in list(src.quote_sources.items()):
        key = next((q for q in smap if normalize(q) == want), None)
        if key is None:
            continue
        rids = list(smap.pop(key))
        if not smap:
            src.quote_sources.pop(product, None)
        had_ids = bool(src.review_ids.get(product))
        # a review whose OTHER quotes still sit in this row keeps its vote here
        still = {r for lst in src.quote_sources.get(product, {}).values()
                 for r in lst}
        src.review_ids[product] = [r for r in src.review_ids.get(product, [])
                                   if r in still or r not in rids]
        if had_ids:
            # reconcile_votes skips cells whose id list went empty, so the
            # count has to be zeroed here or the emptied row keeps ghost votes
            src.votes[product] = len(src.review_ids[product])
        _refresh_quote_sample(src, product)
        if dst is not None:
            for rid in rids:
                dst.add(product, 1, key, [rid], [(rid, key)])
            if dst.review_ids.get(product):
                dst.votes[product] = len(dst.review_ids[product])
            _refresh_quote_sample(dst, product)
        moved += len(rids)
    return moved


def _relocate_canonical(tax: Taxonomy, canon, target_group) -> None:
    """Move `canon` into `target_group`. If the target already holds a
    canonical with the same normalized text, it's that same vote's
    dual-placement twin — absorb it there instead of creating a duplicate
    row (which would also double the vote count)."""
    twin = next((c for c in target_group.canonicals(tax)
                 if c.id != canon.id and normalize(c.text) == normalize(canon.text)),
                None)
    if twin is not None:
        _dedupe_canonical_into(tax, twin, canon)
    else:
        canon.group_id = target_group.id


def _merge_group_into(tax: Taxonomy, source_id: str, target_id: str) -> bool:
    src = tax.groups.get(source_id)
    tgt = tax.groups.get(target_id)
    if not src or not tgt or src.id == tgt.id or src.category != tgt.category:
        return False
    for c in list(src.canonicals(tax)):
        _relocate_canonical(tax, c, tgt)
    if src.usage_category and not tgt.usage_category:
        tgt.usage_category = src.usage_category
    del tax.groups[src.id]
    return True


async def _verify_new_groups(tax: Taxonomy, category: str, new_ids: list[str],
                             llm: LLM) -> list[str]:
    """Self-consistency check: a group created mid-run is re-judged against
    the groups that already existed. Cheap filter against duplicate themes
    accumulating while the taxonomy grows. Returns action descriptions."""
    new_ids = [gid for gid in new_ids if gid in tax.groups]
    existing = [g for g in tax.groups_for(category) if g.id not in set(new_ids)]
    if not new_ids or not existing:
        return []
    # only question new groups that look lexically close to an existing one —
    # obviously novel themes skip the extra call
    suspects = []
    for gid in new_ids:
        name = tax.groups[gid].name
        if any(similarity(name, g.name) >= 0.25 for g in existing):
            suspects.append(gid)
    if not suspects:
        return []

    lines = ["EXISTING GROUPS:"]
    for g in sorted(existing, key=lambda g: -g.total(tax)):
        lines.append(f"  {g.id}: {g.name}  ({g.total(tax)})")
    lines.append("\nNEWLY CREATED GROUPS:")
    for gid in suspects:
        g = tax.groups[gid]
        lines.append(f"  {g.id}: {g.name}")
        for c in g.canonicals(tax):
            lines.append(f'      row: "{c.text}"')
    user = f"Category: {category}\n\n" + "\n".join(lines)
    result = await llm.json_call_async(VERIFY_SYSTEM, user, VERIFY_SCHEMA)

    actions = []
    for d in result.get("decisions", []):
        if d.get("action") != "merge_into":
            continue
        src_id = (d.get("group_id") or "").strip()
        tgt_id = (d.get("target_group_id") or "").strip()
        if src_id not in suspects:
            continue
        src_name = tax.groups[src_id].name if src_id in tax.groups else src_id
        if _merge_group_into(tax, src_id, tgt_id):
            actions.append(f"[{category}] нову групу «{src_name}» влито в "
                           f"«{tax.groups[tgt_id].name}»")
    return actions


CONSOLIDATE_SYSTEM = """\
You are shown the FULL two-level taxonomy of customer review phrases for ONE
category (GROUP = one broad theme worth one bullet point; rows inside the
group carry all the nuance).

The taxonomy was built incrementally, so early batches may have spawned
duplicate themes. Propose the MINIMAL set of fixes:
 1. merges: groups that are the same broad theme — synonyms, rewordings or
    sub-cases of each other (a narrower or reworded version of a benefit
    another group already covers). Pick the strongest group (best name, most
    votes) as keep_group_id.
 2. moves: individual rows sitting in the wrong group. Place a row by its
    DOMINANT angle — the aspect that carries its distinctive content, not a
    feature it merely mentions in passing (a row whose real point is HOW the
    product is applied belongs to the ease-of-use theme even if it also names
    the core benefit; a row whose real point is KEEPING it for an occasion
    belongs to the keep-on-hand theme). Do not move a row out of a theme just
    because it shares a word with another theme.

Do NOT merge groups that are genuinely different benefits/problems/usages,
even if they share words or the same object/component. E.g. "no safety seal
at purchase" and "the container cracks or splits during use" both concern
the container, but name two different defects — keep them separate. Sharing
a component or object name is not evidence of the same theme; the SPECIFIC
failure or benefit must match, not just what it happened to affect. Do NOT
merge rows with each other. A typical category ends up with roughly 10-25
groups. When unsure, change nothing.
All ids must be copied VERBATIM from the taxonomy.
"""

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keep_group_id": {"type": "string"},
                    "merge_group_ids": {"type": "array",
                                        "items": {"type": "string"}},
                },
                "required": ["keep_group_id", "merge_group_ids"],
                "additionalProperties": False,
            },
        },
        "moves": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_id": {"type": "string"},
                    "target_group_id": {"type": "string"},
                },
                "required": ["canonical_id", "target_group_id"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["merges", "moves"],
    "additionalProperties": False,
}


def consolidate_taxonomy(tax: Taxonomy, llm: LLM,
                         progress=None) -> list[str]:
    return run_async(consolidate_taxonomy_async(tax, llm, progress))


async def consolidate_taxonomy_async(tax: Taxonomy, llm: LLM,
                                     progress=None) -> list[str]:
    """Final whole-taxonomy review: one call per category proposes merges of
    duplicate groups and moves of misplaced canonicals. Applied directly to
    the taxonomy; returns human-readable descriptions of what was done.
    Merges/moves never cross categories, so the per-category prompts are
    independent: all calls run concurrently and the results are applied in
    category order."""
    actions: list[str] = []
    cats = [c for c in _domain.active().ids() if len(tax.groups_for(c)) > 3]
    # max_canonicals=None: a `moves` proposal can only target a row the
    # judge actually saw (see _taxonomy_context's docstring) — the old
    # cutoff of 30 silently exempted oversized groups from ever being
    # cleaned up, which is exactly where over-merging accumulates.
    users = {category: (f"Category: {category}\n\nTAXONOMY:\n"
                        f"{_taxonomy_context(tax, category, max_canonicals=None)}")
             for category in cats}
    n_done = 0

    async def call(category: str) -> dict:
        nonlocal n_done
        result = await llm.json_call_async(
            CONSOLIDATE_SYSTEM + _PLACEMENT_RULES + _judge_examples_block(),
            users[category], CONSOLIDATE_SCHEMA)
        n_done += 1
        if progress:
            progress(category, n_done, len(cats))
        return result

    results = await asyncio.gather(*(call(c) for c in cats))

    for category, result in zip(cats, results):
        for m in result.get("merges", []):
            keep_id = (m.get("keep_group_id") or "").strip()
            for src_id in m.get("merge_group_ids", []):
                src_id = (src_id or "").strip()
                src = tax.groups.get(src_id)
                if src is None or keep_id not in tax.groups:
                    continue
                src_name = src.name
                if _merge_group_into(tax, src_id, keep_id):
                    actions.append(f"[{category}] групу «{src_name}» влито в "
                                   f"«{tax.groups[keep_id].name}»")

        for mv in result.get("moves", []):
            cid = (mv.get("canonical_id") or "").strip()
            tgt_id = (mv.get("target_group_id") or "").strip()
            canon = tax.canonicals.get(cid)
            tgt = tax.groups.get(tgt_id)
            if not canon or not tgt or canon.group_id == tgt.id:
                continue
            src = tax.groups.get(canon.group_id)
            if src is None or src.category != tgt.category:
                continue
            text, src_name, tgt_name = canon.text, src.name, tgt.name
            _relocate_canonical(tax, canon, tgt)
            actions.append(f"[{category}] рядок «{text}» перенесено "
                           f"з «{src_name}» у «{tgt_name}»")
        # a move can leave a group empty — drop empty groups
        for g in list(tax.groups_for(category)):
            if not g.canonicals(tax):
                del tax.groups[g.id]
    return actions


# ---------- LLM row-merge pass (sibling rows inside one group) ----------

ROWMERGE_SYSTEM = """\
You clean up ROWS inside a two-level taxonomy of customer review phrases
(GROUP = one broad theme; ROW = one canonical message whose near-identical
customer wordings are merged and votes summed).

The taxonomy was built incrementally, so one MESSAGE is often scattered
across several sibling rows. You get groups with their rows. WITHIN EACH
GROUP, find sets of rows that carry the SAME single message and must become
ONE row.

MERGE rows when they are the same claim about the same aspect:
 - negated complaints: negation form and degree words never split rows —
   "They just don't do it" == "Didn't do it very well" == "have no effect"
   == "It's not working" -> ONE row;
 - synonymous verbs for the SAME failure/benefit about the SAME aspect are
   one message (different words for the same action, same object) -> ONE row;
 - short paraphrases of the same single claim, even with different words ->
   ONE row.
DO NOT merge:
 - different object / part / surface / context: a generic claim vs the same
   claim pinned to a specific object -> keep separate;
 - a meaningful qualifier on one side: a timeframe or degree is content;
   "fast" vs "quickly" vs "instantly" never swap;
 - positive praise tiers never mix: "Works well" vs "Works great" -> keep
   separate rows;
 - a LONG multi-detail descriptive sentence stays its own row even when its
   claim matches a short row;
 - rows from DIFFERENT groups — never propose those.

keep_canonical_id = the row whose wording best names the whole message
(usually the most-voted, most generic customer wording). Copy all ids
VERBATIM from the list. When unsure, do not merge.
"""

ROWMERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keep_canonical_id": {"type": "string"},
                    "merge_canonical_ids": {"type": "array",
                                            "items": {"type": "string"}},
                },
                "required": ["keep_canonical_id", "merge_canonical_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["merges"],
    "additionalProperties": False,
}


def merge_sibling_rows(tax: Taxonomy, llm: LLM, audit: list | None = None,
                       progress=None, max_rows_per_call: int = 120
                       ) -> list[str]:
    return run_async(merge_sibling_rows_async(tax, llm, audit, progress,
                                              max_rows_per_call))


async def merge_sibling_rows_async(tax: Taxonomy, llm: LLM,
                                   audit: list | None = None,
                                   progress=None,
                                   max_rows_per_call: int = 120
                                   ) -> list[str]:
    """LLM row-merge: sibling rows that carry the same message collapse into
    one row. Complements consolidate_rows (deterministic, lexical): only a
    judge can see that "don't stick" == "won't hold" == "no adhesion".
    Every proposal still passes the hard-rule veto (merge_blocked), and
    dual-placement twins are never touched. Groups are independent, so
    small groups are packed together into one call. Each group appears in
    exactly one prompt, so a merge can never change another prompt — all
    prompts are built up front, the calls gathered concurrently, and the
    results applied in (category, batch) order."""
    actions: list[str] = []
    dom = _domain.active()
    jobs: list[tuple[str, dict, str]] = []   # (category, text_count, user)
    per_cat_total: dict[str, int] = {}
    for category in dom.ids():
        if not dom.gated(category):
            continue
        groups = [g for g in tax.groups_for(category)
                  if len(g.canonicals(tax)) >= 2]
        if not groups:
            continue
        # pack groups into calls of <= max_rows_per_call rows; a group
        # larger than the cap gets a call of its own (rows of one group
        # must be judged together)
        batches: list[list] = []
        cur, cur_rows = [], 0
        for g in sorted(groups, key=lambda g: -g.total(tax)):
            n = len(g.canonicals(tax))
            if cur and cur_rows + n > max_rows_per_call:
                batches.append(cur)
                cur, cur_rows = [], 0
            cur.append(g)
            cur_rows += n
        if cur:
            batches.append(cur)

        # rows whose normalized text exists in 2+ groups are dual
        # placements — the same vote deliberately counted twice; merging
        # either copy would double-count or destroy the second theme
        text_count: dict[str, int] = defaultdict(int)
        for g in tax.groups_for(category):
            for c in g.canonicals(tax):
                text_count[normalize(c.text)] += 1

        per_cat_total[category] = len(batches)
        for gb in batches:
            lines = []
            for g in gb:
                lines.append(f"GROUP {g.id}: {g.name}")
                for c in sorted(g.canonicals(tax), key=lambda c: -c.total):
                    lines.append(f'  {c.id}: "{c.text}"  ({c.total})')
            user = f"Category: {category}\n\n" + "\n".join(lines)
            jobs.append((category, text_count, user))

    per_cat_done: dict[str, int] = defaultdict(int)

    async def call(category: str, user: str) -> dict:
        result = await llm.json_call_async(ROWMERGE_SYSTEM, user,
                                           ROWMERGE_SCHEMA)
        per_cat_done[category] += 1
        if progress:
            progress(category, per_cat_done[category],
                     per_cat_total[category])
        return result

    results = await asyncio.gather(*(call(category, user)
                                     for category, _, user in jobs))

    for (category, text_count, _), result in zip(jobs, results):
        for m in result.get("merges", []):
            ids = [(m.get("keep_canonical_id") or "").strip()]
            ids += [(i or "").strip()
                    for i in m.get("merge_canonical_ids", [])]
            cans = [tax.canonicals[i] for i in dict.fromkeys(ids)
                    if i in tax.canonicals]
            cans = [c for c in cans
                    if text_count[normalize(c.text)] == 1]
            if len(cans) < 2:
                continue
            gid = cans[0].group_id
            cans = [c for c in cans if c.group_id == gid]
            if len(cans) < 2:
                continue
            # the most-voted wording names the row (stable sort keeps
            # the judge's keep-choice first among equal vote counts)
            cans.sort(key=lambda c: -c.total)
            keep = cans[0]
            for other in cans[1:]:
                reason = merge_blocked(other.text, keep.text)
                g_to = tax.groups.get(gid)
                if reason is not None:
                    # human feedback loop: exact/near precedents and
                    # rule weights from gate_labels.json may lift the veto
                    blocked, basis = precedents.veto_verdict(
                        category, other.text, keep.text, reason)
                    if blocked:
                        if audit is not None:
                            entry = {
                                "type": "gate_blocked",
                                "category": category,
                                "phrase": other.text, "into": keep.text,
                                "reason": reason,
                                "group": g_to.name if g_to else "?"}
                            if basis:
                                entry["basis"] = basis
                            audit.append(entry)
                        continue
                    if audit is not None:
                        audit.append({
                            "type": "gate_overridden",
                            "category": category,
                            "phrase": other.text, "into": keep.text,
                            "reason": reason, "basis": basis,
                            "group": g_to.name if g_to else "?"})
                actions.append(f"[{category}] рядок «{other.text}» "
                               f"злито в «{keep.text}»")
                if audit is not None and reason is None:
                    audit.append({
                        "type": "row_merge", "category": category,
                        "phrase": other.text, "into": keep.text,
                        "group": g_to.name if g_to else "?"})
                _merge_canonical_into(tax, keep, other)
    return actions


# ---------- split pass: inverse of consolidate_taxonomy ----------
#
# consolidate_taxonomy, merge_sibling_rows and _verify_new_groups only ever
# MERGE groups/rows together. Groups are otherwise only ever created during
# the very first batch-placement pass (group_phrases), before the taxonomy's
# real structure exists yet — so a theme that first pass lumped into an
# early mega-group (e.g. "Hamster Loves It" absorbing a distinct "litter box
# use" theme) had no later pass that could ever pull it back out. This pass
# is that missing inverse: it proposes splitting an OVERSIZED group into 2-4
# real subgroups, or moving individual rows into a SIBLING group that
# already covers their theme.
#
# Placed AFTER consolidate_taxonomy (sees the deduplicated group set) and
# BEFORE reassign_phrases: reassign then replays the WHOLE corpus against
# the enriched taxonomy, so a freshly split group only needs a few seed rows
# here — reassign pulls in every matching phrase from every product/batch on
# its own, no extra machinery needed.

SPLIT_SYSTEM = """\
You are shown ONE oversized group of a customer-review taxonomy (GROUP = one
broad theme worth a single bullet point) — every row it currently holds with
its vote count, the names of its SIBLING groups already in this category (so
you never invent a duplicate of one that exists), and any THEMES already
established elsewhere in this corpus.

Decide whether this group actually bundles MORE THAN ONE distinct selling
point / complaint / usage type / requested change. If so, propose the
MINIMAL split that fixes it:

 - splits: 2-4 NEW subgroups, each a genuinely distinct theme worth its own
   bullet on a product page. List the canonical_ids (copied VERBATIM) that
   belong to each. A subgroup needs at least 2 rows carrying real content —
   never split off a single stray row.
 - moves: individual rows whose theme is ALREADY covered by a SIBLING group
   listed below — give that sibling's group_id instead of creating a
   duplicate-named subgroup.

Leave EVERY row you do not mention exactly where it is — the original group
keeps its name and whatever rows remain.

Do NOT split off:
 - phrasing variants or sub-cases of the same point (that nuance belongs at
   the row level INSIDE one group, not as a separate group);
 - a row that merely mentions a second attribute in passing when its
   dominant angle still matches the group.

ALWAYS split off rows that make an OPPOSITE claim about the same tracked
attribute from rows making the positive claim (e.g. "no scent at all" vs "a
pleasant floral scent" are different, sometimes contradictory, selling
points — never one group), even when the group is otherwise small.

If a THEME PRESENT ELSEWHERE IN THIS CORPUS matches a cluster of rows here,
name your split subgroup to MATCH that theme's wording, so rows from other
categories/products can converge on it later.

When the group really is one coherent theme, return empty splits and moves.
All ids copied VERBATIM from the data below.
"""

SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "splits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "canonical_ids": {"type": "array",
                                     "items": {"type": "string"}},
                },
                "required": ["name", "canonical_ids"],
                "additionalProperties": False,
            },
        },
        "moves": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_id": {"type": "string"},
                    "target_group_id": {"type": "string"},
                },
                "required": ["canonical_id", "target_group_id"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["splits", "moves"],
    "additionalProperties": False,
}


def _apply_split_result(tax: Taxonomy, category: str, group: Group,
                        result: dict) -> list[str]:
    """Apply one group's split/move decisions to the taxonomy. Pure/testable
    without an LLM: `result` is the already-parsed judge answer. Rejects any
    move/split referencing a row that isn't actually still in `group` (e.g.
    an earlier job in the same pass already relocated it) or a subgroup too
    small to be a real theme (< 2 rows or < 3 total votes) — those rows
    simply stay put, no error."""
    actions: list[str] = []
    used_ids: set[str] = set()
    for mv in result.get("moves", []):
        cid = (mv.get("canonical_id") or "").strip()
        tgt = (mv.get("target_group_id") or "").strip()
        c = tax.canonicals.get(cid)
        target = tax.groups.get(tgt)
        if (not c or c.group_id != group.id or not target
                or target.category != category or target.id == group.id):
            continue
        _relocate_canonical(tax, c, target)
        used_ids.add(cid)
        actions.append(f"[{category}] рядок «{c.text}» перенесено з "
                       f"«{group.name}» у «{target.name}»")
    for sp in result.get("splits", []):
        name = (sp.get("name") or "").strip()
        ids = [i for i in sp.get("canonical_ids", []) if i not in used_ids]
        cans = [tax.canonicals[i] for i in ids
               if i in tax.canonicals and tax.canonicals[i].group_id == group.id]
        if not name or len(cans) < 2 or sum(c.total for c in cans) < 3:
            continue
        # a split that reuses an existing group's name is a MOVE into that
        # group, not a second group under the same label; naming the group
        # being split from is a no-op the rows would silently not survive
        existing = _group_named(tax, category, name)
        if existing is not None and existing.id == group.id:
            continue
        new_g = existing or tax.new_group(category, name)
        for c in cans:
            c.group_id = new_g.id
            used_ids.add(c.id)
        actions.append(f"[{category}] «{name}» виділено з «{group.name}» "
                       f"({len(cans)} рядків, "
                       f"{sum(c.total for c in cans)} голосів)")
    return actions


def split_groups(tax: Taxonomy, llm: LLM, progress=None,
                 audit: list | None = None, min_share: float = 0.12,
                 min_rows: int = 25) -> list[str]:
    return run_async(split_groups_async(tax, llm, progress=progress,
                                        audit=audit, min_share=min_share,
                                        min_rows=min_rows))


async def split_groups_async(tax: Taxonomy, llm: LLM, progress=None,
                             audit: list | None = None,
                             min_share: float = 0.12,
                             min_rows: int = 25) -> list[str]:
    """A group is a split candidate when it holds >= min_share of its
    category's total votes, or >= min_rows distinct canonical rows, or
    contains a polarity conflict (similarity.polarity_conflict) regardless
    of size — an "unscented"/"smells amazing" pair is wrong at any size.
    One independent LLM call per candidate group; results are applied in
    job order so output stays deterministic regardless of gather()
    completion order (same pattern as consolidate_taxonomy_async)."""
    actions: list[str] = []
    dom = _domain.active()
    cats = list(dom.ids())
    cat_totals = {cat: sum(g.total(tax) for g in tax.groups_for(cat)) or 1
                 for cat in cats}
    other_themes_by_cat = {
        cat: sorted({og.name for cat2 in cats if cat2 != cat
                    for og in tax.groups_for(cat2) if og.total(tax) >= 5})
        for cat in cats
    }

    jobs: list[tuple[str, Group]] = []
    for cat in cats:
        conflict_gids = {gid for gid, _, _ in polarity_conflicts(tax, cat)}
        for g in tax.groups_for(cat):
            rows = g.canonicals(tax)
            if len(rows) < 2:
                continue
            if (g.id in conflict_gids
                    or g.total(tax) / cat_totals[cat] >= min_share
                    or len(rows) >= min_rows):
                jobs.append((cat, g))

    n_done = 0

    async def call(cat: str, group: Group) -> dict:
        nonlocal n_done
        sibling_lines = [f"  {sg.id}: {sg.name}"
                         for sg in tax.groups_for(cat) if sg.id != group.id]
        row_lines = [f'  {c.id}: "{c.text}"  ({c.total})' for c in
                    sorted(group.canonicals(tax), key=lambda c: -c.total)]
        lines = [f"Category: {cat}", "",
                f"GROUP TO REVIEW: {group.name}  "
                f"(total votes: {group.total(tax)})",
                "ROWS:"] + row_lines
        lines += ["", "SIBLING GROUPS (do not duplicate these themes):"]
        lines += sibling_lines
        if other_themes_by_cat[cat]:
            lines += ["", "THEMES PRESENT ELSEWHERE IN THIS CORPUS:"]
            lines += [f"  - {t}" for t in other_themes_by_cat[cat]]
        result = await llm.json_call_async(SPLIT_SYSTEM + _PLACEMENT_RULES,
                                           "\n".join(lines),
                                           SPLIT_SCHEMA)
        n_done += 1
        if progress:
            progress(cat, n_done, len(jobs))
        return result

    results = await asyncio.gather(*(call(cat, g) for cat, g in jobs))

    for (cat, group), result in zip(jobs, results):
        if group.id not in tax.groups:
            continue   # emptied by an earlier job's moves in this same pass
        group_actions = _apply_split_result(tax, cat, group, result)
        actions.extend(group_actions)
        if audit is not None and group_actions:
            audit.append({"type": "split", "category": cat,
                          "group": group.name, "changes": len(group_actions)})
    return actions


# ---------- final reassignment pass ----------

REASSIGN_SYSTEM = """\
You re-place customer review phrases into a FINAL two-level taxonomy
(GROUP = one broad theme worth one bullet point; CANONICAL = one row inside
a group, near-identical wordings merged, votes summed).

The taxonomy was built incrementally, so early phrases were placed before
the full structure existed. Place each phrase as if the whole taxonomy had
been known from the start. The taxonomy is FINAL — never invent a new group.

For each phrase decide exactly one:
 a) merge into an existing canonical (canonical_id) — ONLY when it is the
    same message with trivial differences: typos, inflection, word order,
    intensifiers (very/really/so/quite), or a praise swap within ONE tier
    (plain tier: good / well / bare "works" / effectiveness idioms like
    "does the trick" or "the right tool for the job"; strong tier: great /
    excellent / outstanding / amazing / wonderful / fantastic / awesome /
    perfect(ly)). Tiers never mix: "works good" == "Works well" but !=
    "Works great". A generic subject, beneficiary or dropped-letter typo
    never blocks a praise merge: "It worked fantastic", "works amazin",
    "worked great for her" all == "Works great". NEGATED complaints merge
    across negation forms, degree words and synonymous failure verbs for the
    SAME aspect: a terse negation and a wordy one of the same failure are one
    row. Short paraphrases of the SAME single claim merge. BUT a meaningfully
    different qualifier ("quickly" vs "instantly", "fast" vs "quickly",
    "worked ok", "for an hour or two"), a different object / part / context
    (a generic claim vs one pinned to a specific object), or a long
    multi-detail sentence vs a short generic row -> NOT the same row;
 b) its own row in an existing group (group_id) — the theme fits, but no
    existing row is the same wording. When in doubt, prefer this over (a).

Choose the group by the phrase's DOMINANT angle — what carries its
distinctive content, not a feature mentioned in passing (a phrase whose real
point is HOW it is applied belongs to the ease-of-use theme; one whose real
point is KEEPING it for an occasion belongs to the keep-on-hand theme). A
complaint blaming a CHANGE (recipe, formula, materials, packaging) for the
product failing at its core job belongs under the theme for THAT failure
(e.g. "doesn't work anymore"), not a standalone "they changed it" group —
unless enough reviews complain about the change itself as the point, not
just as the blamed cause. If one phrase genuinely covers two distinct
themes, also fill second_group_id.
If a quote names two benefits that already have SEPARATE rows in the SAME
group ("great for an emergency or first aid kit" -> the emergencies row and
the first-aid row), place it in one via canonical_id and set
second_canonical_id to the other existing row so the vote counts in both.

ANSWER PROTOCOL (strict): canonical_id / group_id copied VERBATIM from the
taxonomy (c12, g3). To put a phrase into the same row/group as an EARLIER
phrase of THIS batch, set merge_with_phrase / same_group_as_phrase to that
phrase's integer index. Check the [similar: ...] hints first, but they are
hints, not rules. Be consistent: identical inputs must produce identical
decisions.
"""


def reassign_phrases(phrases: list[ExtractedPhrase], tax: Taxonomy, llm: LLM,
                     batch_size: int = 25, progress=None,
                     audit: list | None = None, checkpoint=None) -> list[str]:
    return run_async(reassign_phrases_async(phrases, tax, llm, batch_size,
                                            progress, audit, checkpoint))


async def reassign_phrases_async(phrases: list[ExtractedPhrase],
                                 tax: Taxonomy, llm: LLM,
                                 batch_size: int = 25, progress=None,
                                 audit: list | None = None,
                                 checkpoint=None) -> list[str]:
    """Second grouping pass: replay the WHOLE corpus against the final,
    consolidated taxonomy. First-pass decisions for early batches were made
    against an immature taxonomy; here every phrase is placed with the full
    structure known — this removes the batch-order dependence. Votes and
    quotes are rebuilt from scratch (structure — groups and row texts — is
    kept); rows left without votes and groups left empty are pruned.
    Returns human-readable descriptions of structural changes.

    The taxonomy is FINAL (no new groups), so batch prompts don't depend on
    one another: the [similar: ...] hints come from a pass-start snapshot of
    each category's rows, every prompt is built up front, all calls run
    concurrently, and the results are applied in (category, batch) order —
    ids of rows created during application stay deterministic."""
    by_cat: dict[str, list[ExtractedPhrase]] = defaultdict(list)
    for p in phrases:
        by_cat[p.category].append(p)

    # snapshot the context BEFORE zeroing so row ordering (by votes) and
    # the shown totals still reflect the first pass
    ctx_by_cat = {cat: _taxonomy_context(tax, cat) for cat in by_cat}
    for c in tax.canonicals.values():
        c.votes, c.quotes, c.review_ids = {}, {}, {}

    jobs: list[tuple[str, list, str, str]] = []  # (category, batch, sys, user)
    per_cat_done: dict[str, int] = {}
    per_cat_total: dict[str, int] = {}
    for category, cat_phrases in by_cat.items():
        uniques = _unique_phrases(cat_phrases)
        pending = _deterministic_prepass(tax, category, uniques, audit)
        done = len(uniques) - len(pending)
        per_cat_done[category] = done
        per_cat_total[category] = len(uniques)
        if progress:
            progress(category, done, len(uniques))
        items = [(c.id, c.text, c.group_id)
                 for g in tax.groups_for(category)
                 for c in g.canonicals(tax)]
        # the taxonomy is FINAL and identical for every batch of this
        # category, so fold it into the (cached) system block: all batches
        # after the first read it back as a ~0.1x prompt-cache hit instead of
        # re-billing the same few-thousand-token context as full-price input
        # on every call. Only the per-phrase list stays in the user message.
        sys_cat = (f"{REASSIGN_SYSTEM}{_PLACEMENT_RULES}"
                   f"{_judge_examples_block()}\n"
                   f"FINAL TAXONOMY (category {category}):\n"
                   f"{ctx_by_cat[category]}")
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            plines = []
            for j, b in enumerate(batch):
                total = sum(b["counts"].values())
                line = f'[{j}] "{b["text"]}" (x{total})'
                if b.get("quote_en"):
                    line += f'  [translation: "{b["quote_en"]}"]'
                line += _hint_from_items(items, b.get("quote_en") or b["text"])
                plines.append(line)
            user = (
                f"Category: {category}\n\n"
                f"PHRASES TO PLACE:\n" + "\n".join(plines)
            )
            jobs.append((category, batch, sys_cat, user))

    async def call(category: str, batch: list, sys_cat: str,
                   user: str) -> dict:
        result = await llm.json_call_async(sys_cat, user, GROUPING_SCHEMA)
        per_cat_done[category] += len(batch)
        if progress:
            progress(category, per_cat_done[category],
                     per_cat_total[category])
        return result

    results = await asyncio.gather(*(call(*job) for job in jobs))

    for (category, batch, _, _), result in zip(jobs, results):
        assignments = sorted(result.get("assignments", []),
                             key=lambda a: a.get("phrase_index", 0))
        batch_canon: dict[int, "object"] = {}
        for a in assignments:
            idx = a.get("phrase_index", -1)
            if not 0 <= idx < len(batch):
                continue
            a.pop("new_group_name", None)   # the taxonomy is final
            canon = _apply_assignment(tax, category, batch[idx], a,
                                      batch_canon, audit=audit,
                                      allow_new_group=False)
            if canon is None:
                canon = _fallback_place(tax, category, batch[idx], audit)
            batch_canon[idx] = canon
        for j, b in enumerate(batch):
            if j not in batch_canon:
                _fallback_place(tax, category, b, audit)
        # same rationale as the grouping pass: persist after every applied
        # batch so a crash mid-application loses at most one batch
        if checkpoint:
            checkpoint()

    # rows whose variants were all re-judged elsewhere dissolve
    actions: list[str] = []
    actions.extend(consolidate_rows(tax, audit))
    for c in list(tax.canonicals.values()):
        if c.total == 0:
            g = tax.groups.get(c.group_id)
            if g is not None and audit is not None:
                audit.append({"type": "row_dissolved", "category": g.category,
                              "row": c.text, "group": g.name})
    actions.extend(prune_empty(tax))
    return actions


def _batch_ref(value, batch_canon: dict):
    """Resolve a within-batch phrase reference. Accepts a proper integer
    field, or defensively an id string like '5' / '0_5' that some models
    emit when pointing at a phrase created earlier in the batch."""
    if isinstance(value, int) and value in batch_canon:
        return batch_canon[value]
    if isinstance(value, str) and re.fullmatch(r"\d+(?:_\d+)?", value):
        ref = int(value.split("_")[-1])
        if ref in batch_canon:
            return batch_canon[ref]
    return None


def _apply_assignment(tax: Taxonomy, category: str, b: dict, a: dict,
                      batch_canon: dict, audit: list | None = None,
                      allow_new_group: bool = True):
    canonical_id = (a.get("canonical_id") or "").strip()
    group_id = (a.get("group_id") or "").strip()
    new_group = (a.get("new_group_name") or "").strip()
    usage_cat = (a.get("usage_category") or "").strip()

    canon = tax.canonicals.get(canonical_id)
    if canon is None:
        ref = _batch_ref(a.get("merge_with_phrase", canonical_id or None),
                         batch_canon)
        if ref is not None:
            canon = ref
    # merge veto: the judge sometimes over-merges ("Work well" into "Works
    # great", long sentences into short canonicals). Only HARD rule
    # violations (negation mismatch, different qualifiers, praise-tier mix,
    # long-vs-short) downgrade the merge to a new row in the same group —
    # semantic synonymy ("don't stick" == "won't hold") is the judge's call,
    # vetoing more than that fragments rows into dozens of near-duplicates.
    # relation-keyed categories (e.g. usage/who_recommended) carry relation
    # labels, not verbatim wordings, so the gate doesn't apply there.
    if canon is not None and _domain.active().gated(category):
        reason = merge_blocked(b["text"], canon.text)
        if reason is not None:
            g = tax.groups.get(canon.group_id)
            # human feedback loop: exact/near precedents and rule weights
            # from gate_labels.json may lift the veto — the judge's merge
            # then proceeds and the spot is logged for audit
            blocked, basis = precedents.veto_verdict(
                category, b["text"], canon.text, reason)
            if blocked:
                if audit is not None:
                    entry = {"type": "gate_blocked", "category": category,
                             "phrase": b["text"], "into": canon.text,
                             "reason": reason,
                             "group": g.name if g else "?"}
                    if basis:
                        entry["basis"] = basis
                    audit.append(entry)
                canon = tax.new_canonical(b["text"], canon.group_id)
            elif audit is not None:
                audit.append({"type": "gate_overridden", "category": category,
                              "phrase": b["text"], "into": canon.text,
                              "reason": reason, "basis": basis,
                              "group": g.name if g else "?"})
    if canon is None:
        group = tax.groups.get(group_id)
        if group is None:
            ref = _batch_ref(a.get("same_group_as_phrase", group_id or None),
                             batch_canon)
            if ref is not None:
                group = tax.groups[ref.group_id]
        if group is None:
            name = _group_name_for(category, b, new_group)
            # reuse a group with the same name if the model retyped it
            existing = _group_named(tax, category, name)
            if existing is None and not allow_new_group:
                return None     # reassign pass: caller falls back instead
            group = existing or tax.new_group(category, name, usage_cat)
        if usage_cat and not group.usage_category:
            group.usage_category = usage_cat
        canon = tax.new_canonical(b["text"], group.id)

    # all raw variants in this bucket normalized to the same text, so they
    # share the one translation the bucket carries (see _unique_phrases).
    # b.get(), not b[]: callers that build a minimal `b` by hand (tests,
    # overrides) don't carry a quote_en key at all.
    quote_en = b.get("quote_en")
    trans_map = ({b["text"]: quote_en,
                  **{q: quote_en for raws in b.get("raws", {}).values()
                     for q in raws}}
                 if quote_en else None)

    for product, count in b["counts"].items():
        raw = "; ".join(b["raws"][product][:3])
        canon.add(product, count, raw, b["review_ids"].get(product),
                  b.get("pairs", {}).get(product), trans_map)

    # dual placement: duplicate the canonical into the second group
    second_id = (a.get("second_group_id") or "").strip()
    second_new = (a.get("second_group_new_name") or "").strip()
    second = None
    if second_id and second_id in tax.groups:
        second = tax.groups[second_id]
    elif second_new:
        second = (_group_named(tax, category, second_new)
                  or tax.new_group(category, second_new))
    if second and second.id != canon.group_id:
        twin = next((c for c in second.canonicals(tax)
                     if normalize(c.text) == normalize(canon.text)), None)
        if twin is None:
            twin = tax.new_canonical(b["text"], second.id)
        for product, count in b["counts"].items():
            raw = "; ".join(b["raws"][product][:3])
            twin.add(product, count, raw, b["review_ids"].get(product),
                     b.get("pairs", {}).get(product), trans_map)

    # dual placement into a specific EXISTING row: a quote naming two benefits
    # that already have their own rows (often in the same group, e.g. "great
    # for an emergency or first aid kit" -> the emergencies row AND the first-
    # aid row) is counted in both. Unlike second_group_id this keeps each
    # row's own wording instead of cloning canon's text.
    other = (a.get("second_canonical_id") or "").strip()
    other_text = (a.get("second_canonical_text") or "").strip()
    dst = None
    if other and other in tax.canonicals:
        dst = tax.canonicals[other]
    elif other_text:
        nt = normalize(other_text)
        dst = next((c for c in tax.canonicals.values()
                    if c.group_id == canon.group_id and normalize(c.text) == nt),
                   None) \
            or next((c for c in tax.canonicals.values()
                     if normalize(c.text) == nt), None)
    if dst is not None and dst.id != canon.id:
        # add() dedupes review_ids per-id internally (models.py Canonical.add),
        # so this must run unconditionally like the second_group_id path
        # above — an outer "any id already there -> skip the whole batch"
        # guard would drop every OTHER id in the batch too, permanently
        # (reconcile_votes only reconstructs from ids actually recorded).
        for product, count in b["counts"].items():
            raw = "; ".join(b["raws"][product][:3])
            dst.add(product, count, raw, b["review_ids"].get(product),
                   b.get("pairs", {}).get(product), trans_map)
    return canon


# ---------- deterministic taxonomy repair ----------

def dedupe_group_names(tax: Taxonomy) -> list[str]:
    """Merge groups of one category that carry the SAME name.

    Two same-named groups in one category are the same theme by definition —
    the workbook renders them as two separate blocks under one label, each
    with its own subtotal, which is never what was meant. The placement paths
    reuse an existing name (see _group_named), but a taxonomy can still reach
    the sheet with a collision: built before that check existed, edited by
    hand through the board, or assembled by an override that created a group
    by name. Measured case: "Litter Box / Potty Use" appearing twice in the
    Positive sheet of the Hamster Sand workbook (2026-08-17).

    Names are compared with fold_label, so two groups differing only in
    grammatical number ("Chinchilla" / "Chinchillas") count as the same name
    — one theme, one group. Deterministic and LLM-free. The strongest group
    (most votes, id as tie-break) keeps its id and name; the rest are merged
    into it, and rows carrying the same text are absorbed rather than
    duplicated (see _relocate_canonical). Safe to run repeatedly; runs on
    every write path from excel_writer.write_workbook, BEFORE reconcile_votes
    so the vote counts are rebuilt from the review ids the merge left
    behind."""
    actions: list[str] = []
    for category in _domain.active().ids():
        by_name: dict[str, list] = defaultdict(list)
        for g in tax.groups_for(category):
            by_name[fold_label(g.name)].append(g)
        for dupes in by_name.values():
            if len(dupes) < 2:
                continue
            dupes.sort(key=lambda g: (-g.total(tax), g.id))
            keep = dupes[0]
            for src in dupes[1:]:
                src_name = src.name
                if _merge_group_into(tax, src.id, keep.id):
                    actions.append(f"[{category}] однойменну групу "
                                   f"«{src_name}» злито в одну")
    return actions


def fold_relation_rows(tax: Taxonomy) -> list[str]:
    """Merge rows of a RELATION category whose labels differ only in
    grammatical number ("chinchilla" / "chinchillas", "quail" / "quails").

    A relation row carries a short extractor-written LABEL, not a customer
    wording, so the merge gate and the row-merge pass skip the whole category
    (Category.gated is False for key="relation"). That is right — the gate's
    rules are about praise tiers and qualifiers in verbatim quotes — but it
    left nobody visiting those rows at all, so number variants of one label
    sat side by side with separate subtotals (Hamster Sand 2026-08-17: 20
    rows under "Other Small Pets" for 14 actual species).

    Within one group only: the same label in two groups is a deliberate dual
    placement. The highest-voted wording survives and names the row; votes
    are re-derived from the distinct review ids by _merge_canonical_into, so
    a review present in both rows stays one vote. Deterministic, LLM-free,
    idempotent."""
    actions: list[str] = []
    dom = _domain.active()
    for category in dom.ids():
        if dom.gated(category):
            continue
        for g in tax.groups_for(category):
            buckets: dict[str, list] = defaultdict(list)
            for c in g.canonicals(tax):
                buckets[fold_label(c.text)].append(c)
            for cans in buckets.values():
                if len(cans) < 2:
                    continue
                cans.sort(key=lambda c: (-c.total, c.id))
                keep = cans[0]
                for other in cans[1:]:
                    actions.append(f"[{category}] рядок «{other.text}» злито "
                                   f"в «{keep.text}»")
                    _merge_canonical_into(tax, keep, other)
    return actions


# ---------- vote reconciliation ----------

def reconcile_votes(tax: Taxonomy) -> int:
    """Enforce the core invariant ONE REVIEW = ONE VOTE: every canonical's
    per-product count is set to the number of DISTINCT review ids that landed
    in it. Heals over-counts left by any path that summed votes while deduping
    ids (e.g. merging two rows that share a review). Cells with no review ids
    (legacy data) are left untouched. Returns the number of phantom votes
    removed. Safe to run repeatedly and on every write path."""
    removed = 0
    for c in tax.canonicals.values():
        for product, ids in c.review_ids.items():
            if ids:
                n = len(set(ids))
                cur = c.votes.get(product, 0)
                if cur != n:
                    removed += cur - n
                    c.votes[product] = n
    return removed


# ---------- manual overrides ----------

def apply_overrides(tax: Taxonomy, path: Path) -> None:
    """overrides.json:
    {
      "rename":           {"Old group name": "New group name"},
      "rename_canonical": {"Old phrase text": "New phrase text"},
      "merge_groups":     [["Keep this group", "Merge this in", "And this"]],
      "move_canonical":   {"canonical phrase text": "Target group name"},
      "merge_canonicals": [["Keep this phrase", "Merge this phrase in", "..."]],
      "create_group":     [{"category": "positive", "name": "Litter box use",
                            "rows": ["He designated it as his litter box",
                                     "immediately started using it to potty"]}],
      "dual_place":       [{"quote": "raw review quote naming two benefits",
                            "rows": ["Row A canonical text", "Row B canonical text"]}],
      "move_quote":       [{"quote": "raw review quote sitting in the wrong row",
                            "from": "Row it was grouped into",
                            "to": "Row it belongs to (created if missing)",
                            "group": "USP for the new row (optional)"}],
      "drop_quote":       [{"quote": "raw review quote that is noise",
                            "from": "Row it was grouped into"}],
      "usage_category":   {"Group name": "Coarse bucket band (Usage sheet)",
                           "Another group": ""}
    }
    Applied after grouping on every run — corrections survive reruns.

    move_canonical now AUTO-CREATES the target group when no group of that
    name exists yet in the SAME category as the row being moved (previously
    a typo'd or not-yet-existing target name silently did nothing — a manual
    fix to an over-merged taxonomy had no way to survive the next run).
    Matching is scoped to the moved row's own category — a same-named group
    in a DIFFERENT category is never used as the target, even if the name
    happens to collide.

    create_group is the general tool for the same problem in the other
    direction: hand-carve a new USP (or a wholly new one) out of any
    existing rows in one category, by their exact canonical text. Reuses an
    existing same-named group in that category if one exists, else creates
    it. This is how a human fixes a mega-group the automated split pass
    (grouping.split_groups) missed or under-split, in a way that survives
    every subsequent run.

    usage_category re-bands a group on a relation sheet (the coarse bucket
    shown as a band above the USPs — `subbucket` categories only). Applied
    LAST, after every rule that can rename, merge or CREATE a group, so a
    bucket set on a hand-carved USP still finds its group. An empty value is
    a real instruction ("no bucket of its own" — the sheet falls back to the
    category's `bucket_labels` band), not a no-op.

    dual_place counts ONE review in two canonical rows (typically two rows of
    the same USP group, e.g. a quote naming both "an emergency" and a "first
    aid kit"). The review behind `quote` is located, then its vote/quote/id is
    ensured present in every row in `rows` (deduped by review id, so reruns
    never double-count). Full vote in each row — same semantics as the
    cross-group dual placement the judge does with second_group_id.
    """
    if not path.exists():
        return
    ov = json.loads(path.read_text(encoding="utf-8"))

    def find_group(name: str):
        return next((g for g in tax.groups.values()
                     if normalize(g.name) == normalize(name)), None)

    def find_group_in(name: str, category: str):
        return next((g for g in tax.groups_for(category)
                     if normalize(g.name) == normalize(name)), None)

    def find_canonical(text: str):
        return next((c for c in tax.canonicals.values()
                     if normalize(c.text) == normalize(text)), None)

    def canonical_category(c) -> str:
        g = tax.groups.get(c.group_id)
        return g.category if g else ""

    for old, new in ov.get("rename", {}).items():
        g = find_group(old)
        if g:
            g.name = new

    for old, new in ov.get("rename_canonical", {}).items():
        c = find_canonical(old)
        if c and new.strip():
            c.text = new.strip()

    for names in ov.get("merge_groups", []):
        if len(names) < 2:
            continue
        keep = find_group(names[0])
        if not keep:
            continue
        for other_name in names[1:]:
            other = find_group(other_name)
            if not other or other.id == keep.id:
                continue
            for c in list(other.canonicals(tax)):
                _relocate_canonical(tax, c, keep)
            # same rule as the live merge (_merge_group_into): the bucket of a
            # retired group is not thrown away when the survivor has none —
            # otherwise a merge done on the board loses its band on replay
            if other.usage_category and not keep.usage_category:
                keep.usage_category = other.usage_category
            del tax.groups[other.id]

    for text, target_name in ov.get("move_canonical", {}).items():
        nt = normalize(text)
        for c in [c for c in tax.canonicals.values()
                 if normalize(c.text) == nt]:
            cat = canonical_category(c)
            if not cat:
                continue
            target = find_group_in(target_name, cat)
            if target is None:
                target = tax.new_group(cat, target_name)
            if c.group_id != target.id:
                _relocate_canonical(tax, c, target)

    for spec in ov.get("create_group", []):
        cat = (spec.get("category") or "").strip()
        name = (spec.get("name") or "").strip()
        rows = [t for t in (spec.get("rows") or []) if t and t.strip()]
        if not cat or not name or not rows:
            continue
        target = find_group_in(name, cat)
        if target is None:
            target = tax.new_group(cat, name)
        for row_text in rows:
            nt = normalize(row_text)
            for c in [c for c in tax.canonicals.values()
                     if canonical_category(c) == cat
                     and normalize(c.text) == nt and c.group_id != target.id]:
                _relocate_canonical(tax, c, target)

    for texts in ov.get("merge_canonicals", []):
        if len(texts) < 2:
            continue
        keep = find_canonical(texts[0])
        if not keep:
            continue
        for other_text in texts[1:]:
            other = find_canonical(other_text)
            if not other or other.id == keep.id:
                continue
            _merge_canonical_into(tax, keep, other)

    for spec in ov.get("dual_place", []):
        quote = (spec.get("quote") or "").strip()
        rows = [t for t in (spec.get("rows") or []) if t and t.strip()]
        if not quote or len(rows) < 2:
            continue
        targets = [c for c in (find_canonical(t) for t in rows) if c]
        if len(targets) < 2:
            continue
        # resolve the source review from WITHIN the named rows, so an
        # identical quote elsewhere (e.g. the same words in another category)
        # can never be picked by mistake. Uses quote_sources — the only
        # structure that actually pairs a quote with ITS review — instead of
        # index-matching quotes[]/review_ids[], which are deduped
        # independently and can differ in length (see models.py Canonical).
        want = normalize(quote)
        src = None  # (product, review_id)
        for c in targets:
            for product, smap in c.quote_sources.items():
                for q, rids in smap.items():
                    if rids and (normalize(q) == want or want in normalize(q)):
                        src = (product, rids[0])
                        break
                if src:
                    break
            if src:
                break
        if not src or not src[1]:
            continue
        product, rid = src
        for c in targets:
            if rid in c.review_ids.get(product, []):
                continue    # already counted here — don't double-add
            c.add(product, 1, quote, [rid], [(rid, quote)])

    # LAST: a raw quote the grouping put in the wrong row is pulled out of it
    # and given to another row (or to a row of its own). Runs after every rule
    # that merges or renames rows, so `from`/`to` name the FINAL wordings.
    for spec in ov.get("move_quote", []):
        quote = (spec.get("quote") or "").strip()
        from_text = (spec.get("from") or "").strip()
        to_text = (spec.get("to") or "").strip()
        if not quote or not from_text or not to_text:
            continue
        source = find_canonical(from_text)
        # nothing to move (already replayed, or the row is gone) — never create
        # the destination row on spec alone, that would leave a voteless ghost
        if source is None or find_quote_key(source, quote) is None:
            continue
        cat = canonical_category(source)
        target = next((c for c in tax.canonicals.values()
                       if c.id != source.id and canonical_category(c) == cat
                       and normalize(c.text) == normalize(to_text)), None)
        if target is None:
            gname = (spec.get("group") or "").strip()
            grp = find_group_in(gname, cat) if gname else None
            if grp is None:
                grp = (tax.new_group(cat, gname) if gname
                       else tax.groups.get(source.group_id))
            if grp is None:
                continue
            target = tax.new_canonical(to_text, grp.id)
        move_quote_between(tax, source, target, quote)
        if source.total == 0:
            del tax.canonicals[source.id]
            g = tax.groups.get(source.group_id)
            if g is not None and not g.canonicals(tax):
                del tax.groups[g.id]

    # right after the moves: quotes the user threw out of the analysis. Order
    # matters — a quote that was first moved and then dropped names its NEW row
    # in `from`, which is the row that exists at this point of the replay.
    for spec in ov.get("drop_quote", []):
        quote = (spec.get("quote") or "").strip()
        from_text = (spec.get("from") or "").strip()
        if not quote or not from_text:
            continue
        source = find_canonical(from_text)
        # already replayed, or the row is gone — nothing left to drop
        if source is None or find_quote_key(source, quote) is None:
            continue
        move_quote_between(tax, source, None, quote)
        if source.total == 0:
            del tax.canonicals[source.id]
            g = tax.groups.get(source.group_id)
            if g is not None and not g.canonicals(tax):
                del tax.groups[g.id]

    # LAST of all: the coarse bucket band. Every group-creating rule above has
    # run by now, so a bucket set on a hand-carved USP resolves. "" is applied
    # as-is — clearing a bucket the LLM invented is a correction too.
    for name, bucket in ov.get("usage_category", {}).items():
        g = find_group(name)
        if g is not None:
            g.usage_category = (bucket or "").strip()
