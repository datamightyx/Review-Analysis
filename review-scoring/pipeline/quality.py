"""Deterministic, LLM-free health checks over a finished taxonomy.

Surfaces the exact defect classes a human reviewer would otherwise only
find by reading hundreds of Excel rows by hand — measured against the
Hamster Sand workbook (2026-08-17): a mega-group holding 21% of one
category's votes, a 3-way scent conflict inside one group, and 4 singleton
groups all went unnoticed until the workbook was read line by line. This
module runs the same three checks automatically before every save.
"""
from __future__ import annotations

from collections import defaultdict

from .models import Taxonomy
from . import domain as _domain
from .grouping import polarity_conflicts
from .similarity import fold_label

TOP_GROUP_SHARE = 0.15     # a group holding >=15% of its category is suspect
BIG_GROUP_ROWS = 30        # matches the old consolidate_taxonomy blind spot
SINGLETON_VOTE_MAX = 1
SINGLETON_COUNT_WARN = 3


def workbook_warnings(tax: Taxonomy) -> list[str]:
    """One human-readable line per issue found, ready to print or log
    before save_workbook. Empty list = nothing flagged. Never raises and
    never blocks a save — this is a visibility aid, not a gate."""
    warnings: list[str] = []
    dom = _domain.active()
    for cat in dom.ids():
        groups = tax.groups_for(cat)
        if not groups:
            continue
        cat_total = sum(g.total(tax) for g in groups) or 1
        # same-named groups: grouping.dedupe_group_names merges them on the
        # way into the workbook, so this line explains a group count that
        # dropped between the board and the saved file. Keyed on fold_label,
        # exactly like the merge it announces — on normalize() a number
        # variant ("Chinchilla" / "Chinchillas") would be merged silently.
        by_name: dict[str, int] = defaultdict(int)
        for g in groups:
            by_name[fold_label(g.name)] += 1
        for g in groups:
            if by_name[fold_label(g.name)] > 1:
                warnings.append(
                    f"[{cat}] «{g.name}» — {by_name[fold_label(g.name)]} груп "
                    f"з однією назвою; їх буде злито в одну при збереженні")
                by_name[fold_label(g.name)] = 1     # report each name once
        singletons = 0
        for g in groups:
            total = g.total(tax)
            rows = g.canonicals(tax)
            share = total / cat_total
            if share >= TOP_GROUP_SHARE:
                warnings.append(
                    f"[{cat}] «{g.name}» тримає {share:.0%} голосів "
                    f"категорії ({total}/{cat_total}) — перевірте, чи це "
                    f"не мегагрупа з кількома темами")
            if len(rows) > BIG_GROUP_ROWS:
                warnings.append(
                    f"[{cat}] «{g.name}» має {len(rows)} рядків — "
                    f"перевищує межу, яку бачить консолідація")
            if len(rows) == 1 and rows[0].total <= SINGLETON_VOTE_MAX:
                singletons += 1
        if singletons >= SINGLETON_COUNT_WARN:
            warnings.append(
                f"[{cat}] {singletons} груп з 1 рядком і "
                f"≤{SINGLETON_VOTE_MAX} голосом — можливо, шум; "
                f"розгляньте об'єднання чи видалення")
        for gid, a, b in polarity_conflicts(tax, cat):
            g = tax.groups.get(gid)
            gname = g.name if g else gid
            warnings.append(
                f'[{cat}] «{gname}»: протилежні заяви в одній групі — '
                f'"{a}" проти "{b}"')
    return warnings
