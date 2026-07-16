"""Domain profile — makes the pipeline a UNIVERSAL scoring tool.

Everything that used to be hard-coded for the styptic-powder / wound-care
domain (which categories exist, what `relation`/`gist` mean, which Excel
sheet each category feeds, the sheet headers) now lives in a data profile.
The default profile reproduces the original five categories exactly, so
existing product folders behave unchanged; a product folder can drop a
`domain.yaml` to redefine the categories for a completely different product
group without touching code.

Installed as a process-global by run.py / app.py before any pipeline pass —
the same pattern as similarity.set_synonym_families and
precedents.set_gate_precedents. Pipeline modules read the active domain via
`active()`; nothing takes a domain argument, so signatures stay small.

Category traits
---------------
key        "quote"    canonical rows are verbatim customer wordings; the
                      merge gate and row-merge pass apply (positive/negative/
                      improvement).
           "relation" canonical rows are a short relation label the extractor
                      fills (a usage object, a recommender) — not a verbatim
                      wording, so the gate/row-merge never touch them
                      (usage/who_recommended).
name_from  "gist"     new groups are named by the extracted `gist` wish label
                      instead of the phrase text (improvement).
subbucket  True       groups carry a coarse bucket (`usage_category`) shown as
                      a band on a relation-style sheet (usage).

Excel sheets are described by SheetSpec; several categories may feed one
sheet (usage + who_recommended share the "Usage" sheet in the default).
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Category:
    id: str
    key: str = "quote"            # "quote" | "relation"
    name_from: str = ""           # "" | "gist"
    subbucket: bool = False       # relation only: coarse usage_category bucket
    # neutral, product-agnostic extraction guidance shown to the LLM
    extract_rule: str = ""
    # short guidance shown to the grouping/consolidation judge for this
    # category (what a GROUP means here); "" = the generic phrasing
    group_hint: str = ""

    @property
    def is_relation(self) -> bool:
        return self.key == "relation"

    @property
    def gated(self) -> bool:
        """Does the deterministic merge gate / row-merge apply to this
        category? Only verbatim-keyed categories carry real wordings."""
        return self.key == "quote"


@dataclass
class SheetSpec:
    title: str
    layout: str                  # "two_level" | "relation" | "wish"
    categories: list[str]
    headers: list[str]
    widths: list[float]
    # relation layout only: category id -> fallback bucket label for groups
    # that carry no usage_category of their own
    bucket_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Domain:
    categories: list[Category]
    sheets: list[SheetSpec]
    # domain-specific placement few-shots shown to the grouping / reassign /
    # consolidate judges. Generic grammar rules live in the prompt code;
    # these carry the DOMAIN's human-approved readings of ambiguous phrases
    # (measured: without them the judge lumps every phrase naming the core
    # benefit into one mega-group — calibration F1 0.95 -> 0.73). A custom
    # domain.yaml supplies its own or none.
    judge_examples: list[str] = field(default_factory=list)

    # ---- lookups used across the pipeline ----
    def ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.categories)

    def get(self, cat_id: str) -> Category | None:
        return self._by_id.get(cat_id)

    def is_relation(self, cat_id: str) -> bool:
        c = self._by_id.get(cat_id)
        return bool(c and c.is_relation)

    def gated(self, cat_id: str) -> bool:
        c = self._by_id.get(cat_id)
        return bool(c and c.gated)

    def name_from(self, cat_id: str) -> str:
        c = self._by_id.get(cat_id)
        return c.name_from if c else ""

    def has_subbucket(self, cat_id: str) -> bool:
        c = self._by_id.get(cat_id)
        return bool(c and c.subbucket)

    @property
    def _by_id(self) -> dict[str, Category]:
        return {c.id: c for c in self.categories}


# --------------------------------------------------------------------------
# Default profile — the original five categories, behaviour-identical.
# --------------------------------------------------------------------------

def default_domain() -> Domain:
    return Domain(
        categories=[
            Category(
                id="positive", key="quote",
                extract_rule=(
                    "a product property, benefit, or praise WITH content "
                    "(a concrete quality, outcome, or advantage). Generic "
                    "effectiveness such as \"they work well\" or \"did the "
                    "job\" IS included — it forms an \"Effective\" bucket."),
                group_hint="a USP statement (one selling point)."),
            Category(
                id="negative", key="quote",
                extract_rule=(
                    "a complaint or drawback with content (a concrete "
                    "failure, defect, or dissatisfaction)."),
                group_hint="a negative statement (one recurring complaint)."),
            Category(
                id="usage", key="relation", subbucket=True,
                extract_rule=(
                    "what or who the product is used for, or how. Fill "
                    "`relation` with the short object of use (the thing, "
                    "subject, situation, or context it was used on/for)."),
                group_hint="a usage type."),
            Category(
                id="improvement", key="quote", name_from="gist",
                extract_rule=(
                    "the customer suggests how to improve the product. Fill "
                    "`gist` with a 2-4 word label of the wish."),
                group_hint="an improvement type (one requested change)."),
            Category(
                id="who_recommended", key="relation",
                extract_rule=(
                    "ONLY when a specific type of person is named as having "
                    "recommended the product. Fill `relation` with who."),
                group_hint="who recommended the product."),
        ],
        sheets=[
            SheetSpec(
                title="Positive", layout="two_level", categories=["positive"],
                headers=["USP statement", "The reviews", "Product",
                         "Number of votes", "Total by product",
                         "Total by USP"],
                widths=[33.38, 44.0, 16.13, 13.0, 13.0, 17.5]),
            SheetSpec(
                title="Negative", layout="two_level", categories=["negative"],
                headers=["Negative statement", "The reviews", "Product",
                         "Number of votes", "Total by product",
                         "Total by negative"],
                widths=[26.88, 44.88, 18.13, 13.0, 13.0, 13.0]),
            SheetSpec(
                title="Usage", layout="relation",
                categories=["usage", "who_recommended"],
                headers=["Why/How do they take them?", "Relation", "Product",
                         "By product", "By relation", "Overall"],
                widths=[32.63, 34.88, 19.5, 17.25, 15.63, 13.0],
                bucket_labels={"who_recommended": "Who recommended",
                               "usage": "Other"}),
            SheetSpec(
                title="Improvement", layout="wish",
                categories=["improvement"],
                headers=["Product name", "Improvement", "Review sample",
                         "Votes", "Overall"],
                widths=[33.75, 55.88, 59.75, 13.0, 13.0]),
        ],
    # the battle-tested examples from calibration against the human-made
    # reference workbook (first-aid domain). They pin down the DOMINANT
    # ANGLE rule on this domain's most magnetic theme (the core benefit):
        judge_examples=[
            '"I keep it if I get to short to stop bleeding" — the real '
            'point is KEEPING it for the mishap -> the must-have / '
            'keep-on-hand group, NOT the stops-bleeding group it names in '
            'passing.',
            '"just apply a small amount and it stops the bleeding" — the '
            'real point is how little effort application takes -> the '
            'ease-of-use group, not stops-bleeding.',
            'a phrase praising the handy container or size stays in the '
            'handy-size group, not in must-have.',
        ],
    )


# --------------------------------------------------------------------------
# Loading a per-product profile (domain.yaml in the product folder).
# --------------------------------------------------------------------------

def _category_from_dict(d: dict) -> Category:
    return Category(
        id=str(d["id"]).strip(),
        key=str(d.get("key", "quote")).strip() or "quote",
        name_from=str(d.get("name_from", "")).strip(),
        subbucket=bool(d.get("subbucket", False)),
        extract_rule=str(d.get("extract_rule", "")).strip(),
        group_hint=str(d.get("group_hint", "")).strip(),
    )


def _sheet_from_dict(d: dict) -> SheetSpec:
    return SheetSpec(
        title=str(d["title"]),
        layout=str(d.get("layout", "two_level")),
        categories=[str(c) for c in d.get("categories", [])],
        headers=[str(h) for h in d.get("headers", [])],
        widths=[float(w) for w in d.get("widths", [])],
        bucket_labels={str(k): str(v)
                       for k, v in (d.get("bucket_labels") or {}).items()},
    )


def domain_from_dict(data: dict) -> Domain:
    cats = [_category_from_dict(c) for c in data.get("categories", [])]
    sheets = [_sheet_from_dict(s) for s in data.get("sheets", [])]
    if not cats or not sheets:
        raise ValueError("domain.yaml must define non-empty "
                         "'categories' and 'sheets'")
    # examples are the profile's own: a custom domain without judge_examples
    # runs on the generic rules only (never inherits another domain's)
    examples = [str(e).strip() for e in (data.get("judge_examples") or [])
                if str(e).strip()]
    return Domain(categories=cats, sheets=sheets, judge_examples=examples)


def load_domain(folder: Path | None = None) -> Domain:
    """Load `<folder>/domain.yaml` if present, else the default profile.
    A malformed profile falls back to the default with the error surfaced by
    the caller via the returned flag is NOT used here — we raise instead so a
    typo never silently reverts a custom domain."""
    if folder is not None:
        p = Path(folder) / "domain.yaml"
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return domain_from_dict(data)
    return default_domain()


def write_default_yaml(path: Path) -> None:
    """Write a commented domain.yaml skeleton (the default profile) so a user
    can customise categories for a new product group."""
    dom = default_domain()
    data = {
        "categories": [
            {"id": c.id, "key": c.key, **({"name_from": c.name_from} if c.name_from else {}),
             **({"subbucket": True} if c.subbucket else {}),
             "extract_rule": c.extract_rule,
             **({"group_hint": c.group_hint} if c.group_hint else {})}
            for c in dom.categories
        ],
        "sheets": [
            {"title": s.title, "layout": s.layout, "categories": s.categories,
             "headers": s.headers, "widths": s.widths,
             **({"bucket_labels": s.bucket_labels} if s.bucket_labels else {})}
            for s in dom.sheets
        ],
        # replace these with YOUR domain's ambiguous-phrase readings (or
        # delete the key): which group a phrase belongs to when it names the
        # core benefit in passing but its real point is something else
        "judge_examples": dom.judge_examples,
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


# --------------------------------------------------------------------------
# Active domain — thread-local (contextvars), not a plain module global.
# Streamlit runs each session's script on its own thread, so a bare global
# here let one session's folder switch silently swap another session's
# mid-run category schema/Excel layout (two users, two product lines, same
# process). ContextVar isolates the value per-thread while keeping the same
# get/set() call sites used by run.py/app.py/the pipeline modules.
# --------------------------------------------------------------------------

_ACTIVE: contextvars.ContextVar[Domain] = contextvars.ContextVar(
    "active_domain", default=default_domain())


def set_active_domain(dom: Domain | None) -> Domain:
    d = dom or default_domain()
    _ACTIVE.set(d)
    return d


def active() -> Domain:
    return _ACTIVE.get()
