"""Lexical similarity + deterministic merge gate — no heavy dependencies.

sentence-transformers is broken in this environment (numpy 2 vs torch), so
the pre-filter uses a deterministic mix of token-set and character-trigram
Jaccard similarity (optional semantic hints live in embeddings.py). The
similarity score only SHORTLISTS candidates for the LLM judge; merging is
decided by merge_compatible / auto_merge_target, which encode the SOP
threshold exactly ("quickly" vs "instantly" stay separate, praise tiers
never mix, subject fillers never block a praise merge).
"""
from __future__ import annotations

import difflib
import re
import unicodedata

_STOPWORDS = {
    "a", "an", "the", "it", "its", "is", "are", "was", "were", "be", "been",
    "i", "my", "me", "we", "our", "you", "your", "this", "that", "these",
    "they", "them", "those", "she", "he", "her", "him", "his", "hers", "us",
    "to", "of", "for", "and", "or", "in", "on", "at", "with", "so", "very",
    "really", "just", "have", "has", "had", "do", "does", "did", "not",
    "will", "would", "gonna", "going", "am", "s", "re", "ll", "ve",
    "get", "gets", "got", "item", "items", "product", "products",
    "thing", "things", "stuff", "one", "ones", "also", "too", "as", "what",
}

# "doesn't" must surface the negation and "they're" must not leave junk
# tokens after punctuation stripping — expand before normalize()-style
# cleanup (normalize() itself stays untouched: it keys exact-match maps in
# grouping.py, so its output must remain stable across versions)
_CONTRACTIONS = (
    ("’", "'"), ("n't", " not"), ("'re", " are"), ("'ll", " will"),
    ("'ve", " have"), ("'m", " am"), ("'d", " would"),
)

# generic effectiveness idioms are plain-tier praise by meaning: map them to
# the words they stand for so the gate and the similarity score see through
# the wording ("these are the right tool for the job" == "Work well")
_IDIOMS = (
    (re.compile(r"\b(do|does|did|doing|done|doin) the (trick|job)\b"),
     "works well"),
    (re.compile(r"\b(get|gets|got|getting) the job done\b"), "works well"),
    (re.compile(r"\bright tools? for the job\b"), "works well"),
    (re.compile(r"\bdoes (exactly )?what (it|they) (is |are |s )?"
                r"(supposed|meant) to( do)?\b"), "works well"),
    # "quick fix" is a noun phrase (a stopgap), not a speed claim — without
    # this the qualifier check would treat its "quick" as a speed qualifier
    # and block "Great to have for emergency or quick fix" from merging with
    # other emergency-essential wordings
    (re.compile(r"\bquick fix\b"), "fix"),
)

_NEGATIONS = {"no", "not", "never", "none", "nothing", "without", "barely",
              "hardly", "nor", "cannot"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _prepare_flag(text: str) -> tuple[str, bool]:
    """normalize() plus contraction expansion and idiom mapping — the form
    all token/gate logic works on (normalize() itself stays stable: it keys
    exact-match maps). The flag reports whether an idiom fired: an idiom
    only licenses a merge when it IS the whole message, so the gate demands
    zero leftover content words on that side ("did the trick" == "works
    well", but "did the trick to staunch the wound" stays its own row)."""
    t = text.casefold()
    for old, new in _CONTRACTIONS:
        t = t.replace(old, new)
    t = normalize(t)
    fired = False
    for rx, repl in _IDIOMS:
        t, n = rx.subn(repl, t)
        fired = fired or bool(n)
    return t, fired


def _prepare(text: str) -> str:
    return _prepare_flag(text)[0]


def _stem(token: str) -> str:
    """Very light iterative suffix stripping so inflections converge to one
    stem: stops/stopped/stopping -> stop, bleeds/bleeding -> ble(ed),
    quickly -> quick. Over-stemming is fine — stems are only compared to
    other stems produced by the same function."""
    while True:
        for suf in ("ing", "edly", "ied", "ies", "ed", "ly", "es", "s"):
            if token.endswith(suf) and len(token) - len(suf) >= 3:
                token = token[:-len(suf)]
                break
        else:
            break
    if len(token) > 3 and token[-1] == token[-2]:
        token = token[:-1]  # stopp -> stop
    return token


def _tokens(text: str) -> set[str]:
    return {_stem(t) for t in _prepare(text).split() if t not in _STOPWORDS}


def _trigrams(text: str) -> set[str]:
    s = " " + _prepare(text) + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# degree words that add no content — interchangeable per the SOP merge rule
_INTENSIFIERS = {
    "very", "really", "so", "quite", "extremely", "pretty", "super",
    "surprisingly", "truly", "incredibly", "absolutely", "totally",
    "definitely", "certainly",
}

def _stems(words: list[str]) -> frozenset[str]:
    return frozenset(_stem(w) for w in words)

# Praise tiers: wordings inside ONE tier are interchangeable ("works good" ==
# "works well"; "works great" == "worked fantastic" == "work perfectly"), but
# tiers never mix ("works well" != "works great" — user's explicit
# threshold). perfectly/flawlessly are STRONG praise (user's 2026-07-06
# decision, supersedes the old "perfectly is its own word" rule).
_PRAISE_TIERS = (
    _stems(["good", "well"]),
    _stems(["great", "excellent", "outstanding", "amazing", "wonderful",
            "fantastic", "awesome", "phenomenal", "terrific", "superb",
            "perfect", "perfectly", "flawless", "flawlessly", "marvelous",
            "brilliant"]),
)

# qualifier words carry real meaning and never swap for anything, not even a
# synonym ("fast" != "quickly", "worked ok" != "worked well"); a merge is
# blocked whenever the two phrases carry DIFFERENT qualifier sets
_QUALIFIER_STEMS = _stems([
    "fast", "quick", "quickly", "instant", "instantly", "immediate",
    "immediately", "rapid", "rapidly", "promptly", "slow", "slowly",
    "away", "overnight", "week", "day", "hour", "minute", "second",
    "month", "year", "forever",
    "ok", "okay", "alright", "decent", "fine",
])

# in the bare-works case ("Handy and work" vs "Work well") only contentless
# evaluative fillers may ride along — a real context word keeps the row
# separate ("works on cats", "works when clipping dogs nails")
_FORGIVABLE_FILLERS = _stems(["handy", "nice"])

# Under mutual negation, degree words stop being content: "Didn't stick very
# well", "Doesn't even stick", "no stick at all", "won't stay on properly"
# are all the same complaint as "They just don't stick". These residues (plus
# negation words themselves and praise stems) are stripped before comparing
# two negated phrases. Only applies when BOTH sides are negated — in positive
# phrases "even"/"properly" may carry meaning.
_NEG_STRIP = _stems(["even", "all", "enough", "properly", "proper",
                     "anymore", "longer", "whatsoever", "much"]) | \
             _stems(sorted(_NEGATIONS))

# Synonym families for the both-negated comparison. The pipeline is a
# UNIVERSAL tool, so no domain vocabulary lives in code: by default the list
# is empty and domain synonymy ("don't stick" == "won't hold" == "no
# adhesion") is the LLM judge's call — the prompts state the general rule
# and merge_blocked() lets such merges through for any product line.
# A product line may still declare families it wants merged
# DETERMINISTICALLY (no judge, stable across reruns) in config.yaml:
#   synonym_families:
#     - [stick, sticky, stickiness, stuck, adhere, adhesion, adhesive, hold, stay]
# Words are given in base form (regular inflections are stemmed away at
# compare time; irregular forms like "stuck"/"held" must be listed).
# Families apply ONLY between two negated phrases: in positive phrases
# "holds" may praise containment, not adhesion.
_SYN_FAMILIES: tuple[frozenset[str], ...] = ()
_SYN_LOOKUP: dict[str, int] = {}


def set_synonym_families(families) -> None:
    """Install per-run synonym families (from config.yaml). Called by
    run.py/app.py before any grouping; families=None/[] clears them."""
    global _SYN_FAMILIES, _SYN_LOOKUP
    fams = []
    for fam in families or ():
        words = [w for w in fam if isinstance(w, str) and w.strip()]
        if len(words) >= 2:
            fams.append(_stems(words))
    _SYN_FAMILIES = tuple(fams)
    _SYN_LOOKUP = {s: i for i, f in enumerate(_SYN_FAMILIES) for s in f}


def _neg_collapse(stems: set[str]) -> set[str]:
    """Both-negated comparison form: drop negation/degree/praise residue,
    fold synonym-family members into one marker."""
    out = set()
    for t in stems:
        if t in _NEG_STRIP or _praise_tier(t) is not None:
            continue
        fam = _SYN_LOOKUP.get(t)
        out.add(f"~fam{fam}" if fam is not None else t)
    return out


def _praise_tier(stem: str) -> int | None:
    """Tier lookup with typo tolerance: review quotes drop letters
    ("amazin", "outstandin", "perfectl") and those must still count as
    their full praise word."""
    for i, tier in enumerate(_PRAISE_TIERS):
        if stem in tier:
            return i
    if len(stem) >= 4:
        for i, tier in enumerate(_PRAISE_TIERS):
            for w in tier:
                if difflib.SequenceMatcher(None, stem, w).ratio() >= 0.8:
                    return i
    return None


def _content_tokens(text: str) -> set[str]:
    return {_stem(t) for t in _prepare(text).split()
            if t not in _STOPWORDS and t not in _INTENSIFIERS}


def _has_work(stems: set[str]) -> bool:
    return any(t == "work" or
               (len(t) >= 4 and
                difflib.SequenceMatcher(None, t, "work").ratio() >= 0.8)
               for t in stems)


def merge_compatible(a: str, b: str) -> bool:
    """Deterministic gate for canonical-level merges: True only when the two
    wordings are the same message per the SOP threshold. Same message means:
    identical up to typos, inflection, word order, intensifiers, subject /
    beneficiary fillers, and a praise-word swap within ONE tier. Encodes:
     - "stop bleed quick" == "It stops the bleeding quickly"
     - "works good" == "works well", but "works well" != "works great"
     - "quickly" != "instantly", "fast" != "quickly", "worked ok" != "well"
     - "The incision worked fantastic" == "Works great" (subject filler)
     - "Didn't stick very well" == "They just don't stick" == "have no
       stick to them" (mutual negation collapses degree words and
       adhesion-verb synonyms: stick/hold/adhere/stay)
     - negation on one side only always blocks."""
    (pa, idiom_a), (pb, idiom_b) = _prepare_flag(a), _prepare_flag(b)
    neg_a = bool(set(pa.split()) & _NEGATIONS)
    neg_b = bool(set(pb.split()) & _NEGATIONS)
    if neg_a != neg_b:
        return False
    ta = {_stem(t) for t in pa.split()
          if t not in _STOPWORDS and t not in _INTENSIFIERS}
    tb = {_stem(t) for t in pb.split()
          if t not in _STOPWORDS and t not in _INTENSIFIERS}
    if not ta or not tb:
        return True
    if {t for t in ta if t in _QUALIFIER_STEMS} != \
            {t for t in tb if t in _QUALIFIER_STEMS}:
        return False                    # a meaningful qualifier differs
    if neg_a and neg_b:
        # two NEGATED phrases are the same complaint when, after dropping
        # negation/degree/praise residue and folding synonym families, the
        # same content remains: "Doesn't even stick to the skin" == "Would
        # not hold on skin" == "Cannot stick well to the skin", while
        # "Dressing don't stick" keeps its extra content word ("dressing")
        # and falls through to the strict logic / the judge
        ca, cb = _neg_collapse(ta), _neg_collapse(tb)
        if ca and ca == cb:
            return True
    extra_a, extra_b = sorted(ta - tb), sorted(tb - ta)
    # pair off near-identical stems (typos: "grate" ~ "great")
    for x in list(extra_a):
        for y in extra_b:
            if difflib.SequenceMatcher(None, x, y).ratio() >= 0.8:
                extra_a.remove(x)
                extra_b.remove(y)
                break
    tiers_a = {_praise_tier(t) for t in extra_a} - {None}
    tiers_b = {_praise_tier(t) for t in extra_b} - {None}
    rest_a = [t for t in extra_a if _praise_tier(t) is None]
    rest_b = [t for t in extra_b if _praise_tier(t) is None]
    if len(tiers_a) > 1 or len(tiers_b) > 1:
        return False                    # mixed praise tiers on one side
    if tiers_a and tiers_b and tiers_a != tiers_b:
        return False                    # swap allowed only within one tier
    if not rest_a and not rest_b:
        if tiers_a and tiers_b:
            return True                 # same-tier praise swap
        # one-sided extra praise: bare "works" may absorb plain praise
        # ("works well") but not strong praise ("works great")
        return (tiers_a | tiers_b) != {1}
    # leftover real content words are forgivable ONLY in the works-pattern:
    # both phrases assert "<something> works <praise>", where the subject /
    # beneficiary is not the message ("The incision worked fantastic",
    # "This worked great for her" == "Works great")
    if _has_work(ta) and _has_work(tb):
        if (idiom_a and rest_a) or (idiom_b and rest_b):
            return False    # an idiom must be the whole message on its side
        full_a = {_praise_tier(t) for t in ta} - {None}
        full_b = {_praise_tier(t) for t in tb} - {None}
        leftover = len(rest_a) + len(rest_b)
        if full_a and full_a == full_b and len(full_a) == 1 and leftover <= 2:
            return True
        if (not full_a or not full_b) and (full_a | full_b) <= {0} \
                and leftover <= 1 \
                and all(t in _FORGIVABLE_FILLERS for t in rest_a + rest_b):
            return True                 # "Handy and work" == "Work well"
    return False


def merge_blocked(a: str, b: str) -> str | None:
    """Hard-rule veto for JUDGE-proposed canonical merges; returns the
    violated rule (for the audit) or None when the merge may proceed.

    merge_compatible answers "is this certainly the same message?" and is
    used to merge WITHOUT the judge — it must stay strict. This function
    answers the opposite question — "does this merge certainly violate the
    SOP threshold?" — because the judge sees semantics the lexical gate
    can't ("don't stick" == "won't hold", "great for emergencies" ==
    "essential part of my emergency kit"). Blocking judge merges on
    anything less than a hard rule causes the row fragmentation the user
    reported (30 sibling rows saying "doesn't stick")."""
    if merge_compatible(a, b):
        return None
    pa, pb = _prepare(a), _prepare(b)
    neg_a = bool(set(pa.split()) & _NEGATIONS)
    neg_b = bool(set(pb.split()) & _NEGATIONS)
    if neg_a != neg_b:
        return "негація лише з одного боку"
    ta = {_stem(t) for t in pa.split()
          if t not in _STOPWORDS and t not in _INTENSIFIERS}
    tb = {_stem(t) for t in pb.split()
          if t not in _STOPWORDS and t not in _INTENSIFIERS}
    if not ta or not tb:
        return None
    if {t for t in ta if t in _QUALIFIER_STEMS} != \
            {t for t in tb if t in _QUALIFIER_STEMS}:
        return "різні кваліфікатори (fast/quickly/ok/таймфрейм)"
    # a long multi-detail sentence never dissolves into a short generic row —
    # detailed wordings are content material and must survive as rows
    if abs(len(ta) - len(tb)) >= 4:
        return "довге описове речення проти короткого рядка"
    if not (neg_a and neg_b):
        # positive praise tiers never mix ("Works well" != "Works great");
        # under mutual negation praise words are residue, not tiers
        tiers_a = {_praise_tier(t) for t in ta} - {None}
        tiers_b = {_praise_tier(t) for t in tb} - {None}
        if tiers_a and tiers_b and tiers_a != tiers_b:
            return "різні рівні похвали"
        # pure works+praise phrases carry no other content that could anchor
        # a semantic merge — for them the strict gate (which just said no)
        # is the whole rule: "they work" never absorbs "Works great"
        def _pure_praise(ts: set[str]) -> bool:
            return all(_praise_tier(t) is not None or _has_work({t})
                       or t in _FORGIVABLE_FILLERS for t in ts)
        if _pure_praise(ta) and _pure_praise(tb):
            return "різні рівні похвали"
        # a KNOWN praise word on one side against a word the tier list has
        # never seen on the other ("Wraps well" vs "wraps nicely": "well" is
        # tier 0, "nicely" is not, so tiers_a/tiers_b above don't even see a
        # conflict) must not slip through for free — only exact tier
        # membership licenses a praise-word swap, never an unvetted lookalike
        extra_a, extra_b = sorted(ta - tb), sorted(tb - ta)
        for x in list(extra_a):
            for y in extra_b:
                if difflib.SequenceMatcher(None, x, y).ratio() >= 0.8:
                    extra_a.remove(x)
                    extra_b.remove(y)
                    break
        # NOTE: _FORGIVABLE_FILLERS is NOT excluded here on purpose — that
        # list ("handy", "nice") only excuses a filler riding along with the
        # bare "work" idiom (merge_compatible's separate, narrower check);
        # reusing it here would let "nicely" (stem "nice") ride free as an
        # unvetted stand-in for "well" in ANY phrase, defeating this rule
        # for the exact case it exists to catch.
        unresolved_a = [t for t in extra_a if _praise_tier(t) is None]
        unresolved_b = [t for t in extra_b if _praise_tier(t) is None]
        if (tiers_a and unresolved_b) or (tiers_b and unresolved_a):
            return "похвальне слово поза визнаним рівнем (потребує підтвердження)"
    return None


def similarity(a: str, b: str) -> float:
    """0..1 lexical similarity: token overlap weighted over trigram overlap."""
    return 0.6 * _jaccard(_tokens(a), _tokens(b)) + \
           0.4 * _jaccard(_trigrams(a), _trigrams(b))


def auto_merge_target(text: str, items: list[tuple[str, str]]
                      ) -> tuple[str, str] | None:
    """Deterministic pre-merge: the best existing canonical `text` can merge
    into WITHOUT asking the judge, or None. Requires merge_compatible plus a
    strong signal that this is the works-praise pattern, identical content,
    or high lexical overlap — borderline cases still go to the judge."""
    ta = _content_tokens(text)
    if not ta:
        return None
    best, best_score = None, -1.0
    for cid, ctext in items:
        tb = _content_tokens(ctext)
        if not tb:
            continue
        s = similarity(text, ctext)
        strong = ta == tb or (_has_work(ta) and _has_work(tb)) or s >= 0.5
        if not strong or not merge_compatible(text, ctext):
            continue
        if s > best_score:
            best, best_score = (cid, ctext), s
    return best


def top_candidates(text: str, items: list[tuple[str, str]], k: int = 3,
                   min_sim: float = 0.30) -> list[tuple[str, str, float]]:
    """items: (id, text). Returns up to k most similar (id, text, score)."""
    scored = [(i, t, similarity(text, t)) for i, t in items]
    scored = [s for s in scored if s[2] >= min_sim]
    scored.sort(key=lambda s: -s[2])
    return scored[:k]
