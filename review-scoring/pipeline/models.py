"""Data model for the review-scoring pipeline.

Two-level structure mirroring the reference workbook:
  Group (USP statement / Negative statement / Usage type / Improvement type)
    -> Canonical (one row in "The reviews" column; near-identical phrasings merged)
       -> votes per product
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CATEGORIES = ("positive", "negative", "usage", "improvement", "who_recommended")


def product_key(stem: str) -> str:
    """Collapse a PDF file stem to a stable per-PRODUCT key by stripping a
    trailing sentiment suffix, so the positive and negative review files of one
    product map to a single product column instead of two pseudo-products:
        B0FFT5KR8L_positive / B0FFT5KR8L_negative      -> B0FFT5KR8L
        B0FB86T1PJ_ppositive                           -> B0FB86T1PJ
        B0D4PHX83N_positive +negative                  -> B0D4PHX83N
    A stem with no sentiment suffix (a normal product name) is returned as-is.
    """
    key = re.sub(r"[ _]+\+?[ _]*(?:pp?ositive|negative)\b.*$", "", stem, flags=re.I)
    return key.strip() or stem


@dataclass
class Review:
    review_id: str
    product: str           # short product name from products.yaml
    author: str
    rating: float | None
    title: str
    date: str              # ISO yyyy-mm-dd, "" if unknown
    variant: str
    body: str


@dataclass
class ExtractedPhrase:
    """One customer quote tagged by the extraction LLM."""
    quote: str
    category: str          # one of CATEGORIES
    product: str
    review_id: str
    relation: str = ""     # usage only: body part / animal / situation
    gist: str = ""         # improvement only: short label of the wish


@dataclass
class Canonical:
    id: str
    text: str              # the representative customer phrase (verbatim)
    group_id: str
    votes: dict[str, int] = field(default_factory=dict)   # product -> count
    quotes: dict[str, list[str]] = field(default_factory=dict)  # product -> raw variants
    review_ids: dict[str, list[str]] = field(default_factory=dict)  # product -> source review ids

    @property
    def total(self) -> int:
        return sum(self.votes.values())

    def add(self, product: str, count: int, raw: str,
            review_ids: list[str] | None = None) -> None:
        self.votes[product] = self.votes.get(product, 0) + count
        self.quotes.setdefault(product, [])
        if raw not in self.quotes[product]:
            self.quotes[product].append(raw)
        if review_ids:
            ids = self.review_ids.setdefault(product, [])
            for rid in review_ids:
                if rid not in ids:
                    ids.append(rid)


@dataclass
class Group:
    id: str
    category: str          # positive / negative / usage / improvement / who_recommended
    name: str              # USP statement, negative statement, usage type, improvement type
    usage_category: str = ""   # usage only: top-level bucket, e.g. "Dogs use"

    def canonicals(self, tax: "Taxonomy") -> list[Canonical]:
        return [c for c in tax.canonicals.values() if c.group_id == self.id]

    def total(self, tax: "Taxonomy") -> int:
        return sum(c.total for c in self.canonicals(tax))


@dataclass
class Taxonomy:
    groups: dict[str, Group] = field(default_factory=dict)
    canonicals: dict[str, Canonical] = field(default_factory=dict)
    _next_id: int = 1

    def new_group(self, category: str, name: str, usage_category: str = "") -> Group:
        gid = f"g{self._next_id}"
        self._next_id += 1
        g = Group(id=gid, category=category, name=name.strip(), usage_category=usage_category)
        self.groups[gid] = g
        return g

    def new_canonical(self, text: str, group_id: str) -> Canonical:
        cid = f"c{self._next_id}"
        self._next_id += 1
        c = Canonical(id=cid, text=text.strip(), group_id=group_id)
        self.canonicals[cid] = c
        return c

    def groups_for(self, category: str) -> list[Group]:
        return [g for g in self.groups.values() if g.category == category]

    def remap_products(self, key_fn=product_key) -> int:
        """Rewrite every canonical's product keys through `key_fn`, merging
        entries that collapse to the same key (union quotes/review_ids; votes
        summed — call reconcile_votes afterwards to re-derive one-per-review).
        Returns how many product cells were renamed. Used to fold a product's
        positive & negative PDFs into a single column."""
        renamed = 0
        for c in self.canonicals.values():
            prods = set(c.votes) | set(c.quotes) | set(c.review_ids)
            v: dict[str, int] = {}
            q: dict[str, list[str]] = {}
            r: dict[str, list[str]] = {}
            for p in prods:
                k = key_fn(p)
                if k != p:
                    renamed += 1
                for rid in c.review_ids.get(p, []):
                    r.setdefault(k, [])
                    if rid not in r[k]:
                        r[k].append(rid)
                for quote in c.quotes.get(p, []):
                    q.setdefault(k, [])
                    if quote not in q[k]:
                        q[k].append(quote)
                if p in c.votes:
                    v[k] = v.get(k, 0) + c.votes[p]
            c.votes, c.quotes, c.review_ids = v, q, r
        return renamed

    # persistence lives in storage/db_client.py (SQLite); the legacy
    # taxonomy.json reader lives in scripts/migrate_to_sqlite.py
