"""Shared builders for the deterministic-core tests."""
from pipeline.models import Taxonomy, Canonical, ExtractedPhrase


def phrase(quote, category="negative", rid="P:1", product="P"):
    return ExtractedPhrase(quote=quote, category=category, product=product,
                           review_id=rid)


def tax_with(rows):
    """rows: list of (group_name, category, canon_text, {product: [review_ids]}).
    Builds a Taxonomy; votes are set to len(review_ids) per product."""
    tax = Taxonomy()
    groups = {}
    for gname, cat, text, ids in rows:
        g = groups.get((gname, cat))
        if g is None:
            g = tax.new_group(cat, gname)
            groups[(gname, cat)] = g
        c = tax.new_canonical(text, g.id)
        for product, rids in ids.items():
            c.votes[product] = len(rids)
            c.review_ids[product] = list(rids)
            c.quotes[product] = [text]
    return tax


def canon_by_text(tax, text):
    return next(c for c in tax.canonicals.values() if c.text == text)
