"""Board edits (pipeline/manual_edits.py): the taxonomy mutation and the
recorded overrides.json rule must describe the SAME correction.

The shape of every test is: apply the op to a live taxonomy, then replay the
recorded rules on an untouched copy of the pre-op taxonomy and assert both end
up in the same state — that is what makes a manual fix survive the next run.
"""
import copy
import json

import pytest

from pipeline import manual_edits as me
from pipeline.grouping import apply_overrides
from tests.helpers import tax_with, canon_by_text


def snapshot(tax):
    """State that must match after a replay — ids are allowed to differ."""
    out = []
    for g in tax.groups.values():
        rows = sorted((c.text, c.total, tuple(sorted(c.votes.items())))
                      for c in g.canonicals(tax))
        # the usage band is part of the end state: a replay that reproduces the
        # rows but drops the band writes a differently banded Excel sheet
        out.append((g.category, g.name, g.usage_category, rows))
    return sorted(out)


def replay(before, ov, tmp_path):
    """Fresh pre-op taxonomy + recorded rules -> post-op state."""
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(ov, ensure_ascii=False), encoding="utf-8")
    fresh = copy.deepcopy(before)
    apply_overrides(fresh, path)
    return fresh


@pytest.fixture
def tax():
    return tax_with([
        ("Sticks well", "positive", "stays put all day", {"A": ["a1", "a2"]}),
        ("Sticks well", "positive", "adheres firmly", {"B": ["b1"]}),
        ("Good value", "positive", "cheap for the count", {"A": ["a3"]}),
        ("Good value", "positive", "worth the price", {"B": ["b2", "b3"]}),
        ("Falls off", "negative", "peels off in an hour", {"A": ["a4"]}),
    ])


def gid(tax, name):
    return next(g.id for g in tax.groups.values() if g.name == name)


# ---------------------------------------------------------------- move

def test_move_row_replays(tax, tmp_path):
    before, ov = copy.deepcopy(tax), {}
    row = canon_by_text(tax, "adheres firmly")
    me.move_rows(tax, ov, [row.id], gid(tax, "Good value"))

    assert ov["move_canonical"] == {"adheres firmly": "Good value"}
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_move_across_categories_refused(tax):
    row = canon_by_text(tax, "adheres firmly")
    with pytest.raises(me.EditError):
        me.move_rows(tax, {}, [row.id], gid(tax, "Falls off"))


def test_move_stale_id_refused(tax):
    with pytest.raises(me.EditError):
        me.move_rows(tax, {}, ["c999"], gid(tax, "Good value"))


# ---------------------------------------------------------------- merge rows

def test_merge_rows_keeps_target_text(tax, tmp_path):
    before, ov = copy.deepcopy(tax), {}
    keep = canon_by_text(tax, "stays put all day")
    other = canon_by_text(tax, "adheres firmly")
    me.merge_rows(tax, ov, [other.id], keep.id)

    survivor = canon_by_text(tax, "stays put all day")
    assert survivor.votes == {"A": 2, "B": 1}
    assert not any(c.text == "adheres firmly" for c in tax.canonicals.values())
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_merge_rows_with_chosen_wording_replays(tax, tmp_path):
    """The user drops A onto B but picks A's wording as the survivor — the
    rule head must be that wording, else the replay merges the other way."""
    before, ov = copy.deepcopy(tax), {}
    keep = canon_by_text(tax, "stays put all day")
    other = canon_by_text(tax, "adheres firmly")
    me.merge_rows(tax, ov, [other.id], keep.id, keep_text="adheres firmly")

    survivor = canon_by_text(tax, "adheres firmly")
    assert survivor.votes == {"A": 2, "B": 1}
    assert ov["merge_canonicals"] == [["adheres firmly", "stays put all day"]]
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


# ---------------------------------------------------------------- merge USPs

def test_merge_groups_keeping_target_name(tax, tmp_path):
    before, ov = copy.deepcopy(tax), {}
    me.merge_groups(tax, ov, [gid(tax, "Good value")], gid(tax, "Sticks well"))

    assert {g.name for g in tax.groups.values()} == {"Sticks well", "Falls off"}
    assert ov["merge_groups"] == [["Sticks well", "Good value"]]
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_merge_groups_keeping_source_name(tax, tmp_path):
    """Keeping the DRAGGED group's name must not become rename+merge — the
    replay would then hold two groups of that name and merge neither."""
    before, ov = copy.deepcopy(tax), {}
    me.merge_groups(tax, ov, [gid(tax, "Good value")], gid(tax, "Sticks well"),
                    name="Good value")

    assert "rename" not in ov
    assert ov["merge_groups"] == [["Good value", "Sticks well"]]
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_merge_groups_with_brand_new_name(tax, tmp_path):
    """apply_overrides runs `rename` BEFORE `merge_groups`, so the merge rule
    head has to be the new name already."""
    before, ov = copy.deepcopy(tax), {}
    me.merge_groups(tax, ov, [gid(tax, "Good value")], gid(tax, "Sticks well"),
                    name="Sticks well and cheap")

    assert ov["rename"] == {"Sticks well": "Sticks well and cheap"}
    assert ov["merge_groups"] == [["Sticks well and cheap", "Good value"]]
    replayed = replay(before, ov, tmp_path)
    assert {g.name for g in replayed.groups.values()} == {
        "Sticks well and cheap", "Falls off"}
    assert snapshot(replayed) == snapshot(tax)


def test_merge_groups_across_categories_refused(tax):
    with pytest.raises(me.EditError):
        me.merge_groups(tax, {}, [gid(tax, "Falls off")],
                        gid(tax, "Sticks well"))


# ---------------------------------------------------------------- new USP

def test_new_group_from_rows_replays(tax, tmp_path):
    before, ov = copy.deepcopy(tax), {}
    row = canon_by_text(tax, "worth the price")
    me.new_group_from_rows(tax, ov, [row.id], "Price/quality")

    assert canon_by_text(tax, "worth the price").group_id == \
        gid(tax, "Price/quality")
    assert ov["create_group"] == [{"category": "positive",
                                   "name": "Price/quality",
                                   "rows": ["worth the price"]}]
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


# ---------------------------------------------------------------- renames

def test_rename_group_replays(tax, tmp_path):
    before, ov = copy.deepcopy(tax), {}
    me.rename_group(tax, ov, gid(tax, "Sticks well"), "Stays in place")

    assert ov["rename"] == {"Sticks well": "Stays in place"}
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_rename_row_replays(tax, tmp_path):
    before, ov = copy.deepcopy(tax), {}
    row = canon_by_text(tax, "cheap for the count")
    me.rename_row(tax, ov, row.id, "great price per sheet")

    assert ov["rename_canonical"] == {
        "cheap for the count": "great price per sheet"}
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_rename_retargets_earlier_rules(tax, tmp_path):
    """Move a row into a group, then rename that group: the older
    move_canonical rule must follow the rename, or the replay moves the row
    into a group name that no longer exists."""
    before, ov = copy.deepcopy(tax), {}
    row = canon_by_text(tax, "adheres firmly")
    me.move_rows(tax, ov, [row.id], gid(tax, "Good value"))
    me.rename_group(tax, ov, gid(tax, "Good value"), "Best value")

    assert ov["move_canonical"] == {"adheres firmly": "Best value"}
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_row_rename_retargets_merge_rule(tax, tmp_path):
    before, ov = copy.deepcopy(tax), {}
    keep = canon_by_text(tax, "stays put all day")
    other = canon_by_text(tax, "adheres firmly")
    me.merge_rows(tax, ov, [other.id], keep.id)
    me.rename_row(tax, ov, keep.id, "stays put the whole day")

    assert ov["merge_canonicals"] == [["stays put the whole day",
                                       "adheres firmly"]]
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


# ---------------------------------------------------------------- idempotency

def test_replaying_rules_twice_changes_nothing(tax, tmp_path):
    ov = {}
    row = canon_by_text(tax, "adheres firmly")
    me.move_rows(tax, ov, [row.id], gid(tax, "Good value"))
    me.merge_groups(tax, ov, [gid(tax, "Good value")], gid(tax, "Sticks well"),
                    name="Sticky and cheap")

    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(ov, ensure_ascii=False), encoding="utf-8")
    once = snapshot(tax)
    apply_overrides(tax, path)
    apply_overrides(tax, path)
    assert snapshot(tax) == once


# ---------------------------------------------------------------- auto merge

def test_auto_merge_folds_equivalent_rows_only():
    tax = tax_with([
        ("Works", "positive", "They work well", {"A": ["a1"]}),
        ("Works", "positive", "they worked well", {"B": ["b1"]}),
        ("Works", "positive", "cheap for the count", {"B": ["b2"]}),
    ])
    ov = {}
    notes = me.auto_merge_group(tax, ov, gid(tax, "Works"))

    texts = {c.text for c in tax.canonicals.values()}
    assert texts == {"They work well", "cheap for the count"}
    assert len(notes) == 1
    assert ov["merge_canonicals"] == [["They work well", "they worked well"]]


def test_auto_merge_respects_focus():
    tax = tax_with([
        ("Works", "positive", "They work well", {"A": ["a1"]}),
        ("Works", "positive", "they worked well", {"B": ["b1"]}),
        ("Works", "positive", "worth the price", {"B": ["b2"]}),
    ])
    focus = [canon_by_text(tax, "worth the price").id]
    assert me.auto_merge_group(tax, {}, gid(tax, "Works"), focus_ids=focus) == []
    assert len(tax.canonicals) == 3


# ---------------------------------------------------------------- suggestions

def test_suggest_row_moves_finds_lookalikes_elsewhere(tax):
    row = canon_by_text(tax, "worth the price")
    extra = tax.new_canonical("worth the price for sure",
                              gid(tax, "Sticks well"))
    extra.votes["A"] = 1
    hits = me.suggest_row_moves(tax, [row.id])
    assert any(h["row"] == extra.id for h in hits)


def test_suggest_group_merges_pairs_similar_names(tax):
    tax.new_group("positive", "Good value for money")
    pairs = me.suggest_group_merges(tax, "positive")
    names = {tuple(sorted((p["keep_name"], p["other_name"]))) for p in pairs}
    assert ("Good value", "Good value for money") in names


# ---------------------------------------------------------------- move quote
# One RAW review quote leaves a row — into another row, or into a row of its
# own. The votes of the reviews that said exactly that quote go with it.

def quote_tax():
    """rows: canonical text -> {product: {raw quote: [review ids]}}."""
    spec = [
        ("Sticks well", "positive", "stays put all day",
         {"A": {"stays put all day": ["a1"],
                "did not budge in the rain": ["a2"]}}),
        ("Sticks well", "positive", "adheres firmly", {"B": {"adheres firmly": ["b1"]}}),
        ("Good value", "positive", "cheap for the count",
         {"A": {"cheap for the count": ["a3"]}}),
    ]
    from pipeline.models import Taxonomy
    t = Taxonomy()
    groups = {}
    for gname, cat, text, products in spec:
        g = groups.get((gname, cat)) or t.new_group(cat, gname)
        groups[(gname, cat)] = g
        c = t.new_canonical(text, g.id)
        for product, qmap in products.items():
            rids = []
            for q, ids in qmap.items():
                c.quote_sources.setdefault(product, {})[q] = list(ids)
                rids += [r for r in ids if r not in rids]
            c.review_ids[product] = rids
            c.votes[product] = len(rids)
            c.quotes[product] = ["; ".join(list(qmap)[:3])]
    return t


def test_move_quote_into_another_row_replays(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    dst = canon_by_text(tax, "cheap for the count")
    me.move_quote(tax, ov, src.id, "did not budge in the rain",
                  target_row_id=dst.id)

    assert src.votes["A"] == 1                    # a2 left with its quote
    assert dst.votes["A"] == 2                    # a3 + a2
    assert "did not budge in the rain" not in src.quote_sources["A"]
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_move_quote_into_its_own_row_replays(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.move_quote(tax, ov, src.id, "did not budge in the rain",
                  new_text="did not budge in the rain")

    fresh = canon_by_text(tax, "did not budge in the rain")
    assert fresh.votes["A"] == 1
    assert fresh.group_id == src.group_id
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_replaying_move_quote_twice_changes_nothing(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.move_quote(tax, ov, src.id, "did not budge in the rain")

    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(ov, ensure_ascii=False), encoding="utf-8")
    fresh = copy.deepcopy(before)
    apply_overrides(fresh, path)
    once = snapshot(fresh)
    apply_overrides(fresh, path)
    assert snapshot(fresh) == once == snapshot(tax)


def test_moving_the_only_quote_empties_and_drops_the_row(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "adheres firmly")
    dst = canon_by_text(tax, "cheap for the count")
    me.move_quote(tax, ov, src.id, "adheres firmly", target_row_id=dst.id)

    # reconcile_votes leaves emptied cells alone, so the row must be gone here
    assert not any(c.text == "adheres firmly" for c in tax.canonicals.values())
    assert dst.votes["B"] == 1
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_move_quote_survives_the_target_row_being_merged_away(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    dst = canon_by_text(tax, "cheap for the count")
    me.move_quote(tax, ov, src.id, "did not budge in the rain",
                  target_row_id=dst.id)
    # the row that received the quote is later merged into another one
    other = canon_by_text(tax, "adheres firmly")
    me.move_rows(tax, ov, [other.id], dst.group_id)
    me.merge_rows(tax, ov, [dst.id, other.id], dst.id)

    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_move_quote_survives_the_source_row_being_renamed(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.move_quote(tax, ov, src.id, "did not budge in the rain")
    me.rename_row(tax, ov, src.id, "stays put the whole day")

    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_move_quote_across_categories_refused():
    tax = quote_tax()
    src = canon_by_text(tax, "stays put all day")
    neg = tax.new_group("negative", "Falls off")
    row = tax.new_canonical("peels off", neg.id)
    row.votes["A"] = 1
    with pytest.raises(me.EditError):
        me.move_quote(tax, {}, src.id, "did not budge in the rain",
                      target_row_id=row.id)


def test_move_quote_without_a_recorded_source_refused():
    tax = quote_tax()
    src = canon_by_text(tax, "stays put all day")
    with pytest.raises(me.EditError):
        me.move_quote(tax, {}, src.id, "a quote nobody ever said")


# ------------------------------------------------------- bulk quote actions
# The board's checkbox selection: several raw quotes moved / parked / dropped
# in one go. The source row is pruned once at the end, never inside the loop.

def test_move_several_quotes_into_one_row_replays(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    dst = canon_by_text(tax, "cheap for the count")
    me.move_quotes(tax, ov, src.id,
                   ["stays put all day", "did not budge in the rain"],
                   target_row_id=dst.id)

    # the emptied source row is gone, both reviews landed on the target
    assert not any(c.text == "stays put all day" for c in tax.canonicals.values())
    assert dst.votes["A"] == 3
    assert len(ov["move_quote"]) == 2
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_move_several_quotes_each_into_its_own_row_replays(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.move_quotes(tax, ov, src.id,
                   ["stays put all day", "did not budge in the rain"],
                   group_id=canon_by_text(tax, "cheap for the count").group_id,
                   each_own_row=True)

    moved = canon_by_text(tax, "did not budge in the rain")
    assert moved.votes["A"] == 1
    assert moved.group_id == gid(tax, "Good value")
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_parking_quotes_without_usp_replays(tmp_path):
    """«Без USP» is an ordinary group created on demand — the workbook writes
    it like any other, the point is that the votes leave the real USP."""
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.move_quotes(tax, ov, src.id, ["did not budge in the rain"],
                   group_name=me.NO_USP_GROUP, each_own_row=True)

    parked = canon_by_text(tax, "did not budge in the rain")
    assert tax.groups[parked.group_id].name == me.NO_USP_GROUP
    assert tax.groups[parked.group_id].category == "positive"
    assert ov["move_quote"][0]["group"] == me.NO_USP_GROUP
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_parking_reuses_an_existing_no_usp_group(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.move_quotes(tax, ov, src.id, ["did not budge in the rain"],
                   group_name=me.NO_USP_GROUP, each_own_row=True)
    other = canon_by_text(tax, "adheres firmly")
    me.move_quotes(tax, ov, other.id, ["adheres firmly"],
                   group_name=me.NO_USP_GROUP, each_own_row=True)

    parked = [g for g in tax.groups.values() if g.name == me.NO_USP_GROUP]
    assert len(parked) == 1
    assert len(parked[0].canonicals(tax)) == 2
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_drop_quotes_takes_their_votes_with_them(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.drop_quotes(tax, ov, src.id, ["did not budge in the rain"])

    assert src.votes["A"] == 1                     # a2 left with its quote
    assert "did not budge in the rain" not in src.quote_sources["A"]
    assert ov["drop_quote"] == [{"quote": "did not budge in the rain",
                                 "from": "stays put all day"}]
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_dropping_every_quote_removes_the_row(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.drop_quotes(tax, ov, src.id,
                   ["stays put all day", "did not budge in the rain"])

    assert not any(c.text == "stays put all day" for c in tax.canonicals.values())
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_replaying_drop_quote_twice_changes_nothing(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.drop_quotes(tax, ov, src.id, ["did not budge in the rain"])

    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(ov, ensure_ascii=False), encoding="utf-8")
    fresh = copy.deepcopy(before)
    apply_overrides(fresh, path)
    once = snapshot(fresh)
    apply_overrides(fresh, path)
    assert snapshot(fresh) == once == snapshot(tax)


def test_drop_survives_the_source_row_being_renamed(tmp_path):
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.drop_quotes(tax, ov, src.id, ["did not budge in the rain"])
    me.rename_row(tax, ov, src.id, "stays put the whole day")

    assert ov["drop_quote"][0]["from"] == "stays put the whole day"
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_dropping_a_quote_that_was_moved_first_replays(tmp_path):
    """drop_quote replays AFTER move_quote, so a quote first pulled into its
    own row and then thrown out names that NEW row as its source."""
    tax = quote_tax()
    before, ov = copy.deepcopy(tax), {}
    src = canon_by_text(tax, "stays put all day")
    me.move_quote(tax, ov, src.id, "did not budge in the rain")
    parked = canon_by_text(tax, "did not budge in the rain")
    me.drop_quotes(tax, ov, parked.id, ["did not budge in the rain"])

    assert not any(c.text == "did not budge in the rain"
                   for c in tax.canonicals.values())
    assert ov["drop_quote"][0]["from"] == "did not budge in the rain"
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_dropping_a_stale_selection_refused():
    tax = quote_tax()
    src = canon_by_text(tax, "stays put all day")
    with pytest.raises(me.EditError):
        me.drop_quotes(tax, {}, src.id, ["a quote nobody ever said"])


# ---------------------------------------------------------------- usage band

def band_tax():
    """A relation-style category whose groups carry a coarse bucket."""
    tax = tax_with([
        ("Cleaning the grill", "usage", "used it on the grill", {"A": ["a1"]}),
        ("Cleaning the pan", "usage", "wiped the pan with it", {"A": ["a2"]}),
        ("Wiping the table", "usage", "cleaned the table", {"B": ["b1"]}),
    ])
    for name, band in [("Cleaning the grill", "Kitchen"),
                       ("Cleaning the pan", "Kitchen")]:
        next(g for g in tax.groups.values() if g.name == name) \
            .usage_category = band
    return tax


def test_set_usage_category_replays(tmp_path):
    tax = band_tax()
    before, ov = copy.deepcopy(tax), {}
    me.set_usage_category(tax, ov, gid(tax, "Wiping the table"), "Kitchen")

    assert ov["usage_category"] == {"Wiping the table": "Kitchen"}
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_clearing_usage_category_replays(tmp_path):
    """"" is an instruction, not a deleted rule — the band the LLM invented
    has to STAY off after the next run."""
    tax = band_tax()
    before, ov = copy.deepcopy(tax), {}
    me.set_usage_category(tax, ov, gid(tax, "Cleaning the pan"), "")

    assert ov["usage_category"] == {"Cleaning the pan": ""}
    me.prune_overrides(ov)
    assert ov["usage_category"] == {"Cleaning the pan": ""}
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_set_usage_category_is_idempotent_on_replay(tmp_path):
    tax = band_tax()
    before, ov = copy.deepcopy(tax), {}
    me.set_usage_category(tax, ov, gid(tax, "Wiping the table"), "Kitchen")

    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(ov, ensure_ascii=False), encoding="utf-8")
    fresh = copy.deepcopy(before)
    apply_overrides(fresh, path)
    once = snapshot(fresh)
    apply_overrides(fresh, path)
    assert snapshot(fresh) == once == snapshot(tax)


def test_unchanged_usage_category_refused():
    tax = band_tax()
    with pytest.raises(me.EditError):
        me.set_usage_category(tax, {}, gid(tax, "Cleaning the pan"), "Kitchen")


def test_usage_category_follows_a_rename(tmp_path):
    tax = band_tax()
    before, ov = copy.deepcopy(tax), {}
    me.set_usage_category(tax, ov, gid(tax, "Wiping the table"), "Kitchen")
    me.rename_group(tax, ov, gid(tax, "Wiping the table"), "Table wiping")

    assert ov["usage_category"] == {"Table wiping": "Kitchen"}
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_usage_category_survives_a_later_merge(tmp_path):
    """The banded group is merged away; the survivor inherits the band on the
    live taxonomy, so the replay has to inherit it too."""
    tax = band_tax()
    before, ov = copy.deepcopy(tax), {}
    me.merge_groups(tax, ov, [gid(tax, "Cleaning the pan")],
                    gid(tax, "Wiping the table"))

    survivor = next(g for g in tax.groups.values()
                    if g.name == "Wiping the table")
    assert survivor.usage_category == "Kitchen"
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_merge_does_not_overwrite_the_survivors_own_band(tmp_path):
    tax = band_tax()
    me.set_usage_category(tax, {}, gid(tax, "Wiping the table"), "Table")
    before, ov = copy.deepcopy(tax), {}
    me.set_usage_category(tax, ov, gid(tax, "Cleaning the pan"), "Kitchen use")
    me.merge_groups(tax, ov, [gid(tax, "Cleaning the pan")],
                    gid(tax, "Wiping the table"))

    survivor = next(g for g in tax.groups.values()
                    if g.name == "Wiping the table")
    assert survivor.usage_category == "Table"
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_rename_usage_bucket_touches_every_group_of_that_band(tmp_path):
    tax = band_tax()
    before, ov = copy.deepcopy(tax), {}
    note, touched = me.rename_usage_bucket(tax, ov, "usage", "Kitchen",
                                           "Kitchen use")

    assert len(touched) == 2
    assert {g.usage_category for g in tax.groups.values()} == {
        "Kitchen use", ""}
    assert ov["usage_category"] == {"Cleaning the grill": "Kitchen use",
                                    "Cleaning the pan": "Kitchen use"}
    assert snapshot(replay(before, ov, tmp_path)) == snapshot(tax)


def test_rename_usage_bucket_refuses_an_unknown_band():
    tax = band_tax()
    with pytest.raises(me.EditError):
        me.rename_usage_bucket(tax, {}, "usage", "Garage", "Garage use")


def test_usage_buckets_lists_bands_loudest_first():
    tax = band_tax()
    me.set_usage_category(tax, {}, gid(tax, "Wiping the table"), "Table")
    assert me.usage_buckets(tax, "usage") == ["Kitchen", "Table"]


# ---------------------------------------------------------------- undo blob

def test_taxonomy_roundtrip_through_dict(tax):
    clone = me.taxonomy_from_dict(me.taxonomy_to_dict(tax))
    assert snapshot(clone) == snapshot(tax)
    assert clone._next_id == tax._next_id
