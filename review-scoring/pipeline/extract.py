"""Phrase extraction: turn each review into tagged verbatim customer quotes.

Encodes the SOP rules from the review-scoring video: keep the customer's
literal wording, allow one quote to serve two categories, skip contentless
praise, capture usage relations and improvement wishes.
"""
from __future__ import annotations

import asyncio
import difflib
import itertools
import re

from . import domain as _domain
from .llm import LLM, run_async
from .models import Review, ExtractedPhrase
from .similarity import normalize, similarity

# The instructions are product-agnostic: the universal quoting / dual-phrase /
# skip rules are fixed here, while the per-category definitions are generated
# from the active domain profile (see pipeline/domain.py). No product-specific
# vocabulary or example phrases live in this prompt.
_EXTRACT_HEADER = """\
You extract marketing-relevant customer phrases from product reviews for a
"review scoring" analysis. For each review, list every useful phrase.

Rules (follow them strictly):

1. VERBATIM QUOTES. A quote is ONE CONTIGUOUS span copied character-for-
   character from the review title or body. You may trim leading/trailing
   filler but never paraphrase, fix grammar, or reorder words. NEVER stitch
   two separate parts of the review into one quote — if an idea is spread
   over two places, extract two phrases or keep only the stronger span. Never
   echo a phrase from these instructions or their examples — every quote must
   be a span of the review being processed. The literal phrasing is the
   product of this analysis.
2. CATEGORIES per phrase (a phrase can have up to two). Assign each phrase to
   one or more of these categories:
"""

_EXTRACT_FOOTER = """\
3. DISTINCT ASPECTS / NO OVERLAP. Never output two phrases from the SAME
   review where one is contained inside the other. If a sentence bundles two
   distinct points — two restrictions, two benefits, two complaints — joined
   by "or"/"and" or spread across sentences ("Not suitable for kids under 10,
   or any sort of facial cuts"; "The tape was old. Nothing would stick"),
   split it into SEPARATE, non-overlapping phrases, one per point, and do NOT
   also keep the whole sentence. Only when the parts cannot stand alone (a
   single inseparable idea) keep them as one phrase.
4. SKIP contentless text: bare praise with no substance ("Perfect", "A+",
   "As advertised", "Good product" with nothing else), a recommendation with
   no named recommender, shipping/delivery remarks, and price-only remarks
   EXCEPT value statements ("good product for the money").
5. One phrase may legitimately carry two categories at once — if so, give it
   both.
6. If a review has nothing useful, return an empty phrase list for it.
"""


def build_extract_system(dom: _domain.Domain | None = None) -> str:
    dom = dom or _domain.active()
    lines = []
    for c in dom.categories:
        rule = c.extract_rule or "a relevant customer phrase."
        lines.append(f"   - {c.id}: {rule}")
    return _EXTRACT_HEADER + "\n".join(lines) + "\n" + _EXTRACT_FOOTER


def build_extract_schema(dom: _domain.Domain | None = None) -> dict:
    dom = dom or _domain.active()
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review_index": {"type": "integer"},
                        "phrases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "quote": {"type": "string"},
                                    "categories": {
                                        "type": "array",
                                        "items": {"type": "string",
                                                  "enum": list(dom.ids())},
                                    },
                                    "relation": {"type": "string"},
                                    "gist": {"type": "string"},
                                },
                                "required": ["quote", "categories"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["review_index", "phrases"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def extract_phrases(reviews: list[Review], llm: LLM, batch_size: int = 8,
                    progress=None) -> list[ExtractedPhrase]:
    return run_async(extract_phrases_async(reviews, llm, batch_size, progress))


async def extract_phrases_async(reviews: list[Review], llm: LLM,
                                batch_size: int = 8,
                                progress=None) -> list[ExtractedPhrase]:
    """Batches are independent, so all LLM calls run concurrently (capped by
    the shared semaphore in llm.py). Results are applied in batch order, so
    the phrase list is identical to a sequential run's."""
    dom = _domain.active()
    valid_cats = set(dom.ids())
    extract_system = build_extract_system(dom)
    extract_schema = build_extract_schema(dom)
    # Never let one batch span two products: the LLM sees a shared prompt of
    # up to `batch_size` reviews and answers by review_index, so an index
    # mix-up at a product boundary would attach one product's real quote to
    # another product's review_id. Chunking within each product's own run
    # keeps any such mix-up contained to that product.
    batches = []
    for _, group_iter in itertools.groupby(reviews, key=lambda r: r.product):
        group = list(group_iter)
        batches += [group[start:start + batch_size]
                    for start in range(0, len(group), batch_size)]

    done = 0

    async def call(batch: list[Review]) -> dict:
        nonlocal done
        lines = []
        for j, r in enumerate(batch):
            lines.append(
                f"[{j}] rating={r.rating} | title: {r.title}\n{r.body}"
            )
        user = "Reviews:\n\n" + "\n\n".join(lines)
        result = await llm.json_call_async(extract_system, user,
                                           extract_schema)
        done += len(batch)
        if progress:
            progress(done, len(reviews))
        return result

    results = await asyncio.gather(*(call(b) for b in batches))

    phrases: list[ExtractedPhrase] = []
    for batch, result in zip(batches, results):
        for item in result.get("reviews", []):
            idx = item.get("review_index", -1)
            if not 0 <= idx < len(batch):
                continue
            r = batch[idx]
            for p in item.get("phrases", []):
                quote = (p.get("quote") or "").strip()
                if not quote:
                    continue
                for cat in p.get("categories", []):
                    if cat not in valid_cats:
                        continue
                    phrases.append(ExtractedPhrase(
                        quote=quote,
                        category=cat,
                        product=r.product,
                        review_id=r.review_id,
                        relation=(p.get("relation") or "").strip(),
                        gist=(p.get("gist") or "").strip(),
                    ))
    return phrases


# ---------- overlap dedup ----------

_COORDINATORS = {"and", "or", "but", "plus", "also", "nor", "yet"}
_TAIL_STOP = {"a", "an", "the", "to", "of", "is", "are", "was", "were", "be",
              "it", "its", "as", "for", "in", "on", "at", "any", "sort", "kind",
              "that", "this", "so", "very", "really", "just", "with"}


def _words(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower(), re.UNICODE)


def _sublist_start(big: list[str], small: list[str]) -> int | None:
    """Index where `small` occurs as a contiguous sublist of `big`, else None."""
    n, m = len(big), len(small)
    if not m or m >= n:
        return None
    for i in range(n - m + 1):
        if big[i:i + m] == small:
            return i
    return None


_COORD_RE = r"(?:and|or|but|plus|also|nor|yet)"


def _distinct_tail(sup: str, nwords_prefix: int) -> str:
    """If the part of `sup` after its first `nwords_prefix` words opens a
    DISTINCT clause — a coordinator after a comma/semicolon ("... , or X") or a
    new sentence ("... . Nothing would stick") — return that tail cleaned of
    its leading separator and coordinator; else "" (a mere continuation /
    elaboration like "... to take care of X" or a comma-less "... and X")."""
    wmatch = list(re.finditer(r"\w+", sup, re.UNICODE))
    if len(wmatch) <= nwords_prefix:
        return ""
    raw = sup[wmatch[nwords_prefix - 1].end():]
    distinct = (re.match(r"^\s*[,;]\s*" + _COORD_RE + r"\b", raw, re.I)
                or re.match(r"^\s*[.!?;]\s+\w", raw))
    if not distinct:
        return ""
    tail = re.sub(r"^\s*[,;:.!?\-‑–—]+\s*", "", raw)
    m = re.match(r"^" + _COORD_RE + r"\b\s*", tail, re.I)
    if m:
        tail = tail[m.end():]
    return tail.strip()


def dedupe_overlapping(phrases: list[ExtractedPhrase]) -> tuple[
        list[ExtractedPhrase], dict]:
    """Guarantee that within one (product, review, category) no phrase is
    contained inside another — the invariant that stops a single review from
    being counted twice (once as a short clause, once inside the longer
    sentence that quotes it). Universal grammar only, no domain vocabulary.

    For a pair where phrase A's words are a contiguous sublist of B's:
      - if A is a PREFIX of B and B's tail is a coordinated / next-sentence
        clause carrying its own content ("... , or any facial cuts";
        "... . Nothing would stick") -> SPLIT: keep A, replace B with just the
        tail (a distinct second aspect kept as its own phrase);
      - otherwise B merely restates A more verbosely -> DROP B (A is the clean
        core; both carry the same review's single vote).
    Returns (new_phrases, {"dropped": [...], "split": [(old,new), ...]}).
    """
    from collections import defaultdict, deque
    from dataclasses import replace as _dc_replace
    groups: dict[tuple, list[ExtractedPhrase]] = defaultdict(list)
    for p in phrases:
        groups[(p.product, p.review_id, p.category)].append(p)

    remove: set[int] = set()
    add: list[ExtractedPhrase] = []
    log = {"dropped": [], "split": []}
    for plist in groups.values():
        if len(plist) < 2:
            continue
        toks = {id(p): _words(p.quote) for p in plist}
        # shorter phrases first so a subset is seen before its superset.
        # A work QUEUE, not a one-shot sorted() snapshot: a split-off tail
        # appended mid-pass must also get its turn as B, or an overlap
        # between the tail and a third phrase is never caught (both would
        # then survive and double-count that review).
        queue = deque(sorted(plist, key=lambda p: len(toks[id(p)]), reverse=True))
        while queue:
            B = queue.popleft()
            if id(B) in remove:
                continue
            for A in plist:
                if A is B or id(A) in remove:
                    continue
                ta, tb = toks[id(A)], toks[id(B)]
                pos = _sublist_start(tb, ta)
                if pos is None:
                    continue
                tail = _distinct_tail(B.quote, len(ta)) if pos == 0 else ""
                content = [w for w in _words(tail) if w not in _TAIL_STOP]
                if tail and content:
                    new = _dc_replace(B, quote=tail)
                    add.append(new)
                    toks[id(new)] = _words(new.quote)
                    plist.append(new)
                    queue.append(new)
                    log["split"].append((B.quote, new.quote))
                else:
                    log["dropped"].append(B.quote)
                remove.add(id(B))
                break
    if not remove and not add:
        return phrases, log
    # `add` must be filtered by `remove` too — a split-off tail (in `add`)
    # can itself later be found redundant against a third phrase and get
    # its id added to `remove` (see the work-queue above); without this
    # filter it would survive in the output regardless.
    result = ([p for p in phrases if id(p) not in remove] +
              [p for p in add if id(p) not in remove])
    return result, log


# ---------- verbatim validation ----------

REPAIR_SYSTEM = """\
An extraction step pulled a "quote" from a customer review, but the quote is
NOT an exact substring of the review — it was paraphrased, stitched together
from separate parts, or invented. Your job: find the customer's real wording.

Rules:
1. Return substrings copied character-for-character from the review text —
   each one a single contiguous span, punctuation and case included.
2. Prefer the shortest contiguous span that carries the same meaning as the
   intended quote.
3. If the intended meaning is expressed in two separate places, return each
   place as its own item (best match first).
4. If the review contains NOTHING that supports the intended quote, return
   an empty list. Never invent or adjust text.
"""

REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "substrings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["substrings"],
    "additionalProperties": False,
}


def _llm_repair(quote: str, source: str, llm: LLM) -> str:
    """Ask the model for the exact review substring behind a failed quote.
    Of the candidates that verify as real substrings, returns the one most
    similar to the intended quote (a stitched quote yields several spans —
    keep the span carrying most of its meaning), else ""."""
    user = f"Review text:\n{source}\n\nIntended quote:\n{quote}"
    try:
        result = llm.json_call(REPAIR_SYSTEM, user, REPAIR_SCHEMA)
    except Exception:
        return ""
    valid = []
    for cand in result.get("substrings", []):
        cand = (cand or "").strip()
        if cand and normalize(cand) and normalize(cand) in normalize(source):
            valid.append(cand)
    if not valid:
        return ""
    best = max(valid, key=lambda c: (similarity(c, quote), len(c)))
    # re-anchor to the review so punctuation/apostrophes are the original's
    ratio, window = _best_window(best, source)
    return window if ratio >= 0.95 and window else best

def _best_window(quote: str, source: str) -> tuple[float, str]:
    """Best-matching word window of `source` for `quote`.
    Returns (similarity ratio, the ORIGINAL text of that window)."""
    q_norm = normalize(quote)
    words = list(re.finditer(r"\S+", source))
    if not q_norm or not words:
        return 0.0, ""
    n = max(1, len(q_norm.split()))
    norm_words = [normalize(m.group()) for m in words]
    best_ratio, best_span = 0.0, (0, 0)
    for size in range(max(1, n - 2), n + 3):
        for i in range(0, len(words) - size + 1):
            cand = " ".join(w for w in norm_words[i:i + size] if w)
            ratio = difflib.SequenceMatcher(None, q_norm, cand).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (words[i].start(), words[i + size - 1].end())
    return best_ratio, source[best_span[0]:best_span[1]].strip()


def validate_verbatim(phrases: list[ExtractedPhrase],
                      reviews: list[Review], llm: LLM | None = None) -> dict:
    """The SOP requires literal customer wording, so every extracted quote is
    checked against its review. Exact (normalized) substrings pass; close
    paraphrases are REPAIRED to the actual review text; if `llm` is given,
    the rest get one LLM repair attempt (find the real substring — catches
    stitched-together quotes). Quotes that still fail are REMOVED from
    `phrases` so invented text never reaches the taxonomy, and reported.
    Returns {"repaired": [...], "unverified": [...]} (unverified = dropped).
    """
    by_id = {r.review_id: r for r in reviews}
    repaired, unverified = [], []
    dropped: set[int] = set()
    for p in phrases:
        r = by_id.get(p.review_id)
        if r is None:
            continue
        source = f"{r.title}\n{r.body}"
        if normalize(p.quote) and normalize(p.quote) in normalize(source):
            continue
        ratio, window = _best_window(p.quote, source)
        if ratio >= 0.8 and window:
            repaired.append({"was": p.quote, "now": window,
                             "review_id": p.review_id})
            p.quote = window
            continue
        if llm is not None:
            fixed = _llm_repair(p.quote, source, llm)
            if fixed:
                repaired.append({"was": p.quote, "now": fixed,
                                 "review_id": p.review_id, "via": "llm"})
                p.quote = fixed
                continue
        unverified.append({"quote": p.quote, "review_id": p.review_id,
                           "product": p.product, "best_match": window,
                           "ratio": round(ratio, 2)})
        dropped.add(id(p))
    if dropped:
        phrases[:] = [p for p in phrases if id(p) not in dropped]
    return {"repaired": repaired, "unverified": unverified}
