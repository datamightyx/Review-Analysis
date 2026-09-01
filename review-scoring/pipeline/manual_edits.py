"""Manual taxonomy edits for the drag & drop board (app.py, tab «Дошка USP»).

Every board action does TWO things:

1. mutates the live `Taxonomy` (so the board, the Excel and scoring.db show the
   result immediately), reusing grouping.py's merge/relocate helpers — they own
   the one-review-one-vote and dual-placement rules;
2. records an equivalent rule in `overrides.json`, so the same correction is
   replayed after every future pipeline run (including `--fresh`).

Both halves are idempotent: replaying the recorded rules on the already-mutated
taxonomy is a no-op, and replaying them on a freshly grouped taxonomy
reproduces the same end state. `tests/test_manual_edits.py` asserts exactly
that for each operation.

Ordering trap (see grouping.apply_overrides): `rename` is applied BEFORE
`merge_groups`, so a "merge A+B and call the result C" correction may never be
recorded as merge-then-rename — the merge would look for a group name that the
rename already consumed. `merge_groups` therefore records the FINAL name as the
head of the merge rule and only adds a `rename` when that name is brand new.

`usage_category` (the coarse band on a relation sheet) is keyed by group name
like the other group-level rules and replays LAST of all, so it survives every
rename/merge/create above it. An empty value is a real instruction ("no band
of its own"), never a deleted rule — see `record_usage_category`.
"""
from __future__ import annotations

from dataclasses import asdict

from pipeline.models import Taxonomy, Group, Canonical
from pipeline.grouping import (_merge_canonical_into, _dedupe_canonical_into,
                               _relocate_canonical, _merge_group_into,
                               find_quote_key, move_quote_between)
from pipeline.similarity import (normalize, merge_compatible, merge_blocked,
                                 similarity)

# similar enough to be worth showing as a suggestion, but not certain enough
# to merge without the user saying so
SUGGEST_MIN_SIM = 0.42
GROUP_SUGGEST_MIN_SIM = 0.34

# Parking group for quotes pulled out of every real USP: an ordinary group
# (one per category, created on demand) that the workbook writes like any
# other. The point is that those quotes stop inflating a real USP's vote, not
# that they disappear from the report.
NO_USP_GROUP = "Без USP"


class EditError(ValueError):
    """A board operation that can no longer be applied (stale ids, empty
    selection, cross-category drop, ...). Carries a user-facing message."""


# ---------------------------------------------------------------- lookups

def group_of(tax: Taxonomy, canon: Canonical) -> Group | None:
    return tax.groups.get(canon.group_id)


def category_of(tax: Taxonomy, canon: Canonical) -> str:
    g = group_of(tax, canon)
    return g.category if g else ""


def rows_by_id(tax: Taxonomy, ids: list[str]) -> list[Canonical]:
    """Existing canonicals for `ids`, in the given order. Silently drops ids
    that vanished (another tab merged them away) — the caller reports how many
    survived."""
    out: list[Canonical] = []
    for cid in ids:
        c = tax.canonicals.get(cid)
        if c is not None and c not in out:
            out.append(c)
    return out


# ---------------------------------------------------------------- overrides

def _lst(ov: dict, key: str) -> list:
    return ov.setdefault(key, [])


def _map(ov: dict, key: str) -> dict:
    return ov.setdefault(key, {})


def _same(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


def record_move(ov: dict, row_text: str, group_name: str) -> None:
    _map(ov, "move_canonical")[row_text] = group_name


def record_merge_rows(ov: dict, keep_text: str, other_texts: list[str]) -> None:
    others = [t for t in other_texts if not _same(t, keep_text)]
    if not others:
        return
    for rule in _lst(ov, "merge_canonicals"):
        if rule and _same(rule[0], keep_text):
            for t in others:
                if not any(_same(t, x) for x in rule):
                    rule.append(t)
            return
    _lst(ov, "merge_canonicals").append([keep_text] + others)


def record_merge_groups(ov: dict, keep_name: str, retired: list[str]) -> None:
    retired = [n for n in retired if not _same(n, keep_name)]
    if not retired:
        return
    for rule in _lst(ov, "merge_groups"):
        if rule and _same(rule[0], keep_name):
            for n in retired:
                if not any(_same(n, x) for x in rule):
                    rule.append(n)
            return
    _lst(ov, "merge_groups").append([keep_name] + retired)


def record_create_group(ov: dict, category: str, name: str,
                        row_texts: list[str]) -> None:
    for spec in _lst(ov, "create_group"):
        if spec.get("category") == category and _same(spec.get("name", ""), name):
            rows = spec.setdefault("rows", [])
            for t in row_texts:
                if not any(_same(t, x) for x in rows):
                    rows.append(t)
            return
    _lst(ov, "create_group").append(
        {"category": category, "name": name, "rows": list(row_texts)})


def record_usage_category(ov: dict, group_name: str, bucket: str) -> None:
    """Keyed by the group's FINAL name; "" is kept on purpose — it means
    "this group has no bucket of its own", which the sheet renders as the
    category's fallback band."""
    _map(ov, "usage_category")[group_name] = bucket


def record_move_quote(ov: dict, quote: str, from_text: str, to_text: str,
                      group_name: str = "") -> None:
    for spec in _lst(ov, "move_quote"):
        if _same(spec.get("quote", ""), quote) and \
                _same(spec.get("from", ""), from_text):
            spec["to"] = to_text
            if group_name:
                spec["group"] = group_name
            return
    spec = {"quote": quote, "from": from_text, "to": to_text}
    if group_name:
        spec["group"] = group_name
    _lst(ov, "move_quote").append(spec)


def record_drop_quote(ov: dict, quote: str, from_text: str) -> None:
    """A raw quote thrown out of the analysis entirely: its reviews stop
    voting anywhere. Named by (quote, row) like `move_quote`, and replayed
    right after it — a quote first moved and then dropped is recorded against
    its NEW row, which is what exists at that point of the replay."""
    for spec in _lst(ov, "drop_quote"):
        if _same(spec.get("quote", ""), quote) and \
                _same(spec.get("from", ""), from_text):
            return
    _lst(ov, "drop_quote").append({"quote": quote, "from": from_text})


def repoint_quote_refs(ov: dict, old: str, new: str) -> None:
    """`move_quote`/`drop_quote` name their rows by text and replay LAST —
    after the merges. So when a row is absorbed by a merge, the rules pointing
    at its old wording have to follow it, or the correction silently stops
    applying. (The merge rules themselves must keep naming the retired
    wording — that is how they find the row to absorb — so they are left
    alone.)"""
    if _same(old, new):
        return
    for spec in _lst(ov, "move_quote"):
        if _same(spec.get("from", ""), old):
            spec["from"] = new
        if _same(spec.get("to", ""), old):
            spec["to"] = new
    for spec in _lst(ov, "drop_quote"):
        if _same(spec.get("from", ""), old):
            spec["from"] = new


def repoint_group_refs(ov: dict, old: str, new: str) -> None:
    """Rules that TARGET a group by name must follow that group when it is
    renamed or absorbed by a merge. `merge_groups` entries are deliberately
    left alone — their tails name retired groups on purpose, and their heads
    still resolve at their own point in the replay order."""
    if _same(old, new):
        return
    mc = _map(ov, "move_canonical")
    for text, target in list(mc.items()):
        if _same(target, old):
            mc[text] = new
    for spec in _lst(ov, "create_group"):
        if _same(spec.get("name", ""), old):
            spec["name"] = new
    # bucket rules are keyed by group name too. First one wins: when a merge
    # repoints several retired names at the survivor, the survivor's own band
    # must not be overwritten by a group it just absorbed (same precedence as
    # the live merge in grouping._merge_group_into).
    uc = _map(ov, "usage_category")
    moving = [n for n in uc if _same(n, old)]
    taken = any(_same(n, new) for n in uc if not _same(n, old))
    for n in moving:
        value = uc.pop(n)
        if not taken:
            uc[new] = value
            taken = True


def retarget_group_name(ov: dict, old: str, new: str) -> None:
    """A group the user just renamed may already be named by earlier rules —
    rewrite them, otherwise the rename fires first on replay and the older
    rule can no longer find its group."""
    if _same(old, new):
        return
    renames = _map(ov, "rename")
    chained = [k for k, v in renames.items() if _same(v, old)]
    for k in chained:
        renames[k] = new
    if not chained:
        renames[old] = new
    renames.pop(new, None)          # a rename onto itself is dead weight

    for rule in _lst(ov, "merge_groups"):
        for i, n in enumerate(rule):
            if _same(n, old):
                rule[i] = new
    repoint_group_refs(ov, old, new)


def retarget_row_text(ov: dict, old: str, new: str,
                      record: bool = True) -> None:
    """Same idea one level down: rules keyed by a phrase's old wording.

    `record=False` rewrites the references WITHOUT adding a rename rule — used
    when the wording change is a side effect of a merge, where the row is
    supposed to keep the *other* row's original text. Recording a rename there
    would fire before `merge_canonicals` on replay, rename both rows to the
    same text and leave the merge unable to find its second row."""
    if _same(old, new):
        return
    if record:
        renames = _map(ov, "rename_canonical")
        chained = [k for k, v in renames.items() if _same(v, old)]
        for k in chained:
            renames[k] = new
        if not chained:
            renames[old] = new
        renames.pop(new, None)

    for rule in _lst(ov, "merge_canonicals"):
        for i, t in enumerate(rule):
            if _same(t, old):
                rule[i] = new
    mc = _map(ov, "move_canonical")
    for text in list(mc):
        if _same(text, old):
            mc[new] = mc.pop(text)
    for spec in _lst(ov, "create_group"):
        rows = spec.get("rows") or []
        for i, t in enumerate(rows):
            if _same(t, old):
                rows[i] = new
    for spec in _lst(ov, "dual_place"):
        rows = spec.get("rows") or []
        for i, t in enumerate(rows):
            if _same(t, old):
                rows[i] = new
    repoint_quote_refs(ov, old, new)


def prune_overrides(ov: dict) -> None:
    """Drop rules that lost their meaning (a merge list collapsed to one
    entry, a rename onto itself)."""
    for key in ("merge_groups", "merge_canonicals"):
        if key in ov:
            ov[key] = [r for r in ov[key] if len(r) >= 2]
    for key in ("rename", "rename_canonical"):
        if key in ov:
            ov[key] = {k: v for k, v in ov[key].items() if not _same(k, v)}
    if "create_group" in ov:
        ov["create_group"] = [s for s in ov["create_group"] if s.get("rows")]
    if "move_quote" in ov:
        ov["move_quote"] = [
            s for s in ov["move_quote"]
            if s.get("quote") and s.get("from") and s.get("to")
            and not _same(s["from"], s["to"])]
    if "drop_quote" in ov:
        ov["drop_quote"] = [s for s in ov["drop_quote"]
                            if s.get("quote") and s.get("from")]
    for key in list(ov):
        if not ov[key]:
            ov.pop(key)


# ---------------------------------------------------------------- operations
# Each returns a short Ukrainian description of what happened, for the toast.

def move_rows(tax: Taxonomy, ov: dict, row_ids: list[str],
              group_id: str) -> str:
    rows = rows_by_id(tax, row_ids)
    target = tax.groups.get(group_id)
    if not rows or target is None:
        raise EditError("Фрази або групи вже немає — оновіть сторінку.")
    for c in rows:
        if category_of(tax, c) != target.category:
            raise EditError("Не можна переносити фразу між категоріями "
                            f"({category_of(tax, c)} → {target.category}).")
    moved = 0
    for c in rows:
        if c.group_id == target.id:
            continue
        text = c.text
        _relocate_canonical(tax, c, target)
        record_move(ov, text, target.name)
        moved += 1
    if not moved:
        raise EditError("Фрази вже в цій групі.")
    return f"Перенесено фраз: {moved} → «{target.name}»"


def merge_rows(tax: Taxonomy, ov: dict, row_ids: list[str], into_id: str,
               keep_text: str | None = None) -> str:
    keep = tax.canonicals.get(into_id)
    others = [c for c in rows_by_id(tax, row_ids) if c.id != into_id]
    if keep is None or not others:
        raise EditError("Рядка вже немає — оновіть сторінку.")
    for c in others:
        if category_of(tax, c) != category_of(tax, keep):
            raise EditError("Злиття фраз можливе лише в межах однієї категорії.")
    texts = [keep.text] + [c.text for c in others]
    final = (keep_text or keep.text).strip() or keep.text
    for c in others:
        if _same(c.text, keep.text):
            _dedupe_canonical_into(tax, keep, c)
        else:
            _merge_canonical_into(tax, keep, c)
    # the surviving row keeps whichever wording the user picked; the rule head
    # must be that same wording so a replay merges INTO it
    if not _same(final, keep.text):
        retarget_row_text(ov, keep.text, final, record=False)
        keep.text = final
    record_merge_rows(ov, final, [t for t in texts if not _same(t, final)])
    # the absorbed wordings are gone from the taxonomy — anything that will
    # look them up AFTER the merges on replay has to follow them here
    for t in texts:
        repoint_quote_refs(ov, t, final)
    return f"Злито рядків: {len(others) + 1} → «{final}»"


def merge_groups(tax: Taxonomy, ov: dict, source_ids: list[str],
                 keep_id: str, name: str | None = None) -> str:
    keep = tax.groups.get(keep_id)
    sources = [tax.groups[g] for g in source_ids
               if g in tax.groups and g != keep_id]
    if keep is None or not sources:
        raise EditError("Групи вже немає — оновіть сторінку.")
    for g in sources:
        if g.category != keep.category:
            raise EditError("Зливати можна лише USP однієї категорії.")
    orig_keep_name = keep.name
    source_names = [g.name for g in sources]
    final = (name or orig_keep_name).strip() or orig_keep_name

    for g in sources:
        _merge_group_into(tax, g.id, keep.id)

    brand_new = not any(_same(final, n)
                        for n in [orig_keep_name] + source_names)
    if brand_new:
        # rename runs before merge_groups on replay, so the merge rule's head
        # is already the NEW name by the time it is looked up
        retarget_group_name(ov, orig_keep_name, final)
        record_merge_groups(ov, final, source_names)
    else:
        retired = [n for n in [orig_keep_name] + source_names
                   if not _same(n, final)]
        record_merge_groups(ov, final, retired)
    # rules that pointed at a group this merge just retired (a phrase moved
    # into it earlier, a hand-carved USP of that name) now have to point at
    # the surviving one — merge_groups runs BEFORE move_canonical/create_group
    # on replay, so the survivor is what exists by then
    for n in [orig_keep_name] + source_names:
        repoint_group_refs(ov, n, final)
    keep.name = final
    # the band is a property of the group, not a reference to a name: after the
    # merge exactly ONE band survives (grouping._merge_group_into only fills an
    # empty one), so any band rule this merge touched is rewritten to that live
    # value — repointing them one by one would let an absorbed group's band
    # overwrite the survivor's.
    uc = _map(ov, "usage_category")
    stale = [k for k in uc
             if any(_same(k, n)
                    for n in [orig_keep_name, final] + source_names)]
    for k in stale:
        uc.pop(k)
    if stale:
        record_usage_category(ov, final, keep.usage_category)
    return f"Злито USP: {len(sources) + 1} → «{final}»"


def new_group_from_rows(tax: Taxonomy, ov: dict, row_ids: list[str],
                        name: str) -> str:
    rows = rows_by_id(tax, row_ids)
    name = (name or "").strip()
    if not rows:
        raise EditError("Фраз уже немає — оновіть сторінку.")
    if not name:
        raise EditError("Вкажіть назву нової USP.")
    cat = category_of(tax, rows[0])
    if any(category_of(tax, c) != cat for c in rows):
        raise EditError("Усі фрази нової USP мають бути з однієї категорії.")
    target = next((g for g in tax.groups_for(cat) if _same(g.name, name)), None)
    if target is None:
        target = tax.new_group(cat, name)
    texts = [c.text for c in rows]
    for c in rows:
        if c.group_id != target.id:
            _relocate_canonical(tax, c, target)
    record_create_group(ov, cat, target.name, texts)
    return f"Створено USP «{target.name}» з {len(rows)} фраз"


def rename_group(tax: Taxonomy, ov: dict, group_id: str, name: str) -> str:
    g = tax.groups.get(group_id)
    name = (name or "").strip()
    if g is None:
        raise EditError("Групи вже немає — оновіть сторінку.")
    if not name or _same(name, g.name):
        raise EditError("Назва не змінилася.")
    old = g.name
    retarget_group_name(ov, old, name)
    g.name = name
    return f"USP перейменовано: «{old}» → «{name}»"


def usage_buckets(tax: Taxonomy, category: str) -> list[str]:
    """Bucket bands already in use in one category, loudest first — the
    picker's options, so a correction reuses an existing band instead of
    inventing a near-duplicate of it."""
    totals: dict[str, int] = {}
    for g in tax.groups_for(category):
        if g.usage_category:
            totals[g.usage_category] = totals.get(g.usage_category, 0) + g.total(tax)
    return [b for b, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]


def set_usage_category(tax: Taxonomy, ov: dict, group_id: str,
                       bucket: str) -> str:
    """Re-band one USP on a relation sheet. `bucket=""` clears the band — the
    group then falls under the category's fallback label, which is a real
    correction (the LLM invents a bucket for every group it can)."""
    g = tax.groups.get(group_id)
    bucket = (bucket or "").strip()
    if g is None:
        raise EditError("Групи вже немає — оновіть сторінку.")
    if bucket == g.usage_category:
        raise EditError("Usage-група не змінилася.")
    old = g.usage_category
    g.usage_category = bucket
    record_usage_category(ov, g.name, bucket)
    if not bucket:
        return f"«{g.name}»: usage-групу «{old}» знято"
    return (f"«{g.name}»: usage-група {f'«{old}» → ' if old else ''}"
            f"«{bucket}»")


def rename_usage_bucket(tax: Taxonomy, ov: dict, category: str,
                        old: str, new: str) -> tuple[str, list[str]]:
    """Rename a whole band at once — a bucket is shared by several USPs, so
    fixing its wording one group at a time is where a hand correction goes
    half-done. Returns (note, touched group ids)."""
    old, new = (old or "").strip(), (new or "").strip()
    if not old:
        raise EditError("Вкажіть, яку usage-групу перейменувати.")
    if not new:
        raise EditError("Вкажіть нову назву usage-групи.")
    if old == new:
        raise EditError("Назва usage-групи не змінилася.")
    touched = [g for g in tax.groups_for(category)
               if g.usage_category == old]
    if not touched:
        raise EditError(f"Немає USP з usage-групою «{old}».")
    for g in touched:
        g.usage_category = new
        record_usage_category(ov, g.name, new)
    return (f"Usage-групу «{old}» перейменовано на «{new}» "
            f"({len(touched)} USP)", [g.id for g in touched])


def rename_row(tax: Taxonomy, ov: dict, row_id: str, text: str) -> str:
    c = tax.canonicals.get(row_id)
    text = (text or "").strip()
    if c is None:
        raise EditError("Рядка вже немає — оновіть сторінку.")
    if not text or _same(text, c.text):
        raise EditError("Формулювання не змінилося.")
    old = c.text
    retarget_row_text(ov, old, text)
    c.text = text
    return f"Формулювання змінено: «{old}» → «{text}»"


def quote_rows(canon: Canonical) -> list[dict]:
    """Raw review quotes of one row, product by product:
    [{product, quote, votes}] sorted loudest first. Empty for legacy rows that
    were saved before quote_sources existed — those quotes can be displayed
    (from the `quotes` samples) but not moved, because nothing pairs them with
    a review."""
    out: list[dict] = []
    for product, smap in canon.quote_sources.items():
        for quote, rids in smap.items():
            out.append({"product": product, "quote": quote,
                        "votes": len(set(rids))})
    out.sort(key=lambda d: (-d["votes"], d["product"], d["quote"]))
    return out


def _live_quotes(src: Canonical, quotes: list[str]) -> list[str]:
    """The selection, deduped and narrowed to quotes this row still holds a
    source for. A stale pick (another tab moved it away, a legacy row with no
    `quote_sources`) is dropped rather than failing the whole batch."""
    wanted = list(dict.fromkeys(q.strip() for q in quotes if q and q.strip()))
    return [q for q in wanted if find_quote_key(src, q) is not None]


def _target_group(tax: Taxonomy, cat: str, src: Canonical,
                  group_id: str | None, group_name: str) -> Group:
    """USP the new rows go into: by name (created on demand — this is how
    «Без USP» appears), else by id, else the source row's own group."""
    if group_name:
        grp = next((g for g in tax.groups_for(cat) if _same(g.name, group_name)),
                   None)
        return grp or tax.new_group(cat, group_name)
    grp = tax.groups.get(group_id or "") or tax.groups.get(src.group_id)
    if grp is None or grp.category != cat:
        raise EditError("Вкажіть USP тієї самої категорії.")
    return grp


def move_quotes(tax: Taxonomy, ov: dict, row_id: str, quotes: list[str],
                target_row_id: str | None = None, new_text: str | None = None,
                group_id: str | None = None, group_name: str = "",
                each_own_row: bool = False) -> str:
    """Pull raw review quotes out of a row — into another row, into one new row
    together, or each into a row of its own (that is also how «Без USP»
    parking works: own rows inside the `NO_USP_GROUP` group). The reviews that
    said exactly those quotes take their votes with them; reviews the row keeps
    for its other quotes stay.

    A quote is treated as row-wide: if the same wording sits under several
    products, all of them move, because "this quote does not belong in this
    row" is a judgment about the text, not about one product's column.

    The source row is pruned once, at the end — pruning inside the loop would
    delete the row the remaining quotes are still being pulled from."""
    src = tax.canonicals.get(row_id)
    if src is None:
        raise EditError("Рядка вже немає — оновіть сторінку.")
    live = _live_quotes(src, quotes)
    if not live:
        raise EditError("Ці цитати вже перенесено або їх джерело не "
                        "збереглося — оновіть сторінку.")
    cat = category_of(tax, src)
    from_text = src.text
    moved = 0
    targets: list[str] = []

    if target_row_id:
        dst = tax.canonicals.get(target_row_id)
        if dst is None:
            raise EditError("Цільового рядка вже немає — оновіть сторінку.")
        if dst.id == src.id:
            raise EditError("Цитати вже в цьому рядку.")
        if category_of(tax, dst) != cat:
            raise EditError("Переносити цитати можна лише в межах однієї "
                            "категорії.")
        for q in live:
            moved += move_quote_between(tax, src, dst, q)
            record_move_quote(ov, q, from_text, dst.text)
        targets.append(dst.text)
    else:
        grp = _target_group(tax, cat, src, group_id, group_name)
        joint = (new_text or "").strip()
        if not each_own_row and not joint:
            raise EditError("Вкажіть формулювання нового рядка.")
        for q in live:
            text = q if each_own_row else joint
            dst = next((c for c in grp.canonicals(tax)
                        if c.id != src.id and _same(c.text, text)), None)
            if dst is None:
                dst = tax.new_canonical(text, grp.id)
            moved += move_quote_between(tax, src, dst, q)
            record_move_quote(ov, q, from_text, dst.text, grp.name)
            if dst.text not in targets:
                targets.append(dst.text)

    if src.total == 0:
        del tax.canonicals[src.id]
    if len(live) == 1:
        return (f"Цитату «{live[0]}» перенесено в «{targets[0]}» "
                f"({moved} відгук(ів))")
    where = (f"«{targets[0]}»" if len(targets) == 1
             else f"{len(targets)} рядків")
    return f"Перенесено цитат: {len(live)} → {where} ({moved} відгук(ів))"


def move_quote(tax: Taxonomy, ov: dict, row_id: str, quote: str,
               target_row_id: str | None = None, new_text: str | None = None,
               group_id: str | None = None) -> str:
    """One quote — `move_quotes` with the single-quote defaults (an own row is
    worded like the quote itself unless the user retyped it)."""
    return move_quotes(tax, ov, row_id, [quote], target_row_id=target_row_id,
                       new_text=(new_text or quote), group_id=group_id)


def drop_quotes(tax: Taxonomy, ov: dict, row_id: str,
                quotes: list[str]) -> str:
    """Throw raw quotes out of the analysis: the reviews that said exactly
    them stop voting anywhere (`move_quote_between` with no destination). A
    review whose OTHER quotes still sit in the row keeps its vote there."""
    src = tax.canonicals.get(row_id)
    if src is None:
        raise EditError("Рядка вже немає — оновіть сторінку.")
    live = _live_quotes(src, quotes)
    if not live:
        raise EditError("Ці цитати вже прибрано або їх джерело не "
                        "збереглося — оновіть сторінку.")
    from_text = src.text
    lost = 0
    for q in live:
        lost += move_quote_between(tax, src, None, q)
        record_drop_quote(ov, q, from_text)
    if src.total == 0:
        del tax.canonicals[src.id]
    return f"Прибрано цитат: {len(live)} (−{lost} голос(ів))"


# ---------------------------------------------------------------- auto merge

def auto_merge_group(tax: Taxonomy, ov: dict, group_id: str,
                     focus_ids: list[str] | None = None) -> list[str]:
    """Deterministic post-drop cleanup: rows of one group that the strict
    lexical gate calls the same message are merged without asking (this is the
    same `merge_compatible` the pipeline trusts to merge without the LLM
    judge). Everything weaker is left to `suggest_row_merges`.

    `focus_ids` limits merges to pairs touching the rows just dropped, so an
    edit never silently reshuffles the rest of the group."""
    g = tax.groups.get(group_id)
    if g is None:
        return []
    focus = set(focus_ids or [])
    notes: list[str] = []
    while True:
        rows = sorted(g.canonicals(tax), key=lambda c: (-c.total, c.id))
        merged_any = False
        for i, keep in enumerate(rows):
            for other in rows[i + 1:]:
                if focus and keep.id not in focus and other.id not in focus:
                    continue
                if _same(keep.text, other.text):
                    _dedupe_canonical_into(tax, keep, other)
                elif merge_compatible(keep.text, other.text):
                    _merge_canonical_into(tax, keep, other)
                else:
                    continue
                record_merge_rows(ov, keep.text, [other.text])
                repoint_quote_refs(ov, other.text, keep.text)
                notes.append(f"«{other.text}» → «{keep.text}»")
                if other.id in focus:
                    focus.discard(other.id)
                    focus.add(keep.id)
                merged_any = True
                break
            if merged_any:
                break
        if not merged_any:
            return notes


# ---------------------------------------------------------------- suggestions

def suggest_row_merges(tax: Taxonomy, group_id: str,
                       min_sim: float = SUGGEST_MIN_SIM,
                       limit: int = 8) -> list[dict]:
    """Near-duplicate rows inside one group that auto-merge did NOT take
    (blocked by a hard SOP rule, or simply not certain enough). Each carries
    the veto reason so the user sees why it wasn't automatic."""
    g = tax.groups.get(group_id)
    if g is None:
        return []
    rows = sorted(g.canonicals(tax), key=lambda c: -c.total)
    out: list[dict] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            s = similarity(a.text, b.text)
            if s < min_sim:
                continue
            out.append({"keep": a.id, "keep_text": a.text,
                        "other": b.id, "other_text": b.text,
                        "score": round(s, 2),
                        "blocked": merge_blocked(a.text, b.text) or ""})
    out.sort(key=lambda d: -d["score"])
    return out[:limit]


def suggest_row_moves(tax: Taxonomy, row_ids: list[str],
                      min_sim: float = SUGGEST_MIN_SIM,
                      limit: int = 8) -> list[dict]:
    """After a phrase was dragged into a group: other phrases of the same
    category, still sitting elsewhere, that look like the one just moved —
    "перенести теж"."""
    rows = rows_by_id(tax, row_ids)
    if not rows:
        return []
    target_id = rows[0].group_id
    target = tax.groups.get(target_id)
    if target is None:
        return []
    seen: dict[str, dict] = {}
    for moved in rows:
        for c in tax.canonicals.values():
            if c.group_id == target_id or c.id in seen:
                continue
            if category_of(tax, c) != target.category:
                continue
            s = similarity(moved.text, c.text)
            if s < min_sim:
                continue
            src = tax.groups.get(c.group_id)
            seen[c.id] = {"row": c.id, "text": c.text, "score": round(s, 2),
                          "from": src.name if src else "?",
                          "like": moved.text}
    out = sorted(seen.values(), key=lambda d: -d["score"])
    return out[:limit]


def suggest_group_merges(tax: Taxonomy, category: str,
                         min_sim: float = GROUP_SUGGEST_MIN_SIM,
                         limit: int = 8) -> list[dict]:
    """USP pairs whose NAMES look like the same theme — candidates for
    «злити USP»."""
    groups = sorted(tax.groups_for(category), key=lambda g: -g.total(tax))
    out: list[dict] = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            s = similarity(a.name, b.name)
            if s < min_sim:
                continue
            out.append({"keep": a.id, "keep_name": a.name,
                        "other": b.id, "other_name": b.name,
                        "score": round(s, 2)})
    out.sort(key=lambda d: -d["score"])
    return out[:limit]


# ---------------------------------------------------------------- undo state

def taxonomy_to_dict(tax: Taxonomy) -> dict:
    return {"next_id": tax._next_id,
            "groups": [asdict(g) for g in tax.groups.values()],
            "canonicals": [asdict(c) for c in tax.canonicals.values()]}


def taxonomy_from_dict(d: dict) -> Taxonomy:
    tax = Taxonomy()
    tax._next_id = d.get("next_id", 1)
    for g in d.get("groups", []):
        tax.groups[g["id"]] = Group(**g)
    for c in d.get("canonicals", []):
        tax.canonicals[c["id"]] = Canonical(**c)
    return tax
