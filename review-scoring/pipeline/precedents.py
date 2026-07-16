"""Feedback loop for the merge gate: human ✓/✗ verdicts from
gate_labels.json (the «🕵 Перевірити» tab) feed back into future runs.

Every gate veto is resolved through three layers of human evidence,
strongest first:
 1. EXACT precedent — this very (category, phrase, into) pair was already
    labeled: the human decision is replayed deterministically
    (✓ keep -> the veto stands forever; ✗ merge -> the veto is lifted);
 2. NEAR precedent — a labeled pair where the SAME gate rule fired and both
    sides are lexically very close to the current pair: the human decision
    transfers (logged with the matched pair so it can be audited);
 3. RULE WEIGHTS — when one rule's vetoes keep being labeled wrong
    (>= soft_threshold share of >= min_labels labels), the rule becomes
    SOFT: the judge's merge proceeds and the spot is logged as
    "gate_overridden" for the human to audit.

All three layers are GLOBAL across products: load_gate_precedents() pools
every product folder's gate_labels.json (aggregate_all_labels) and builds
one GatePrecedents from the union, so a verdict labeled on one product also
resolves the same (layer 1) or a lexically near (layer 2) pair on any other
product, and the rule weights (layer 3) sum every product's ✓/✗. This is
safe because both precedent layers still gate on matching category AND high
lexical similarity (≥ near_precedent on BOTH sides for layer 2), so an
unrelated product's phrasing simply never matches — only genuinely
recurring wordings transfer. The layer-3 tally is also persisted, human-
readable, to `products/gate_rule_weights.json` (SHARED_WEIGHTS_FILENAME in
the products root — the parent of the product folders); that file is
rebuilt on every load_gate_precedents() and whenever the GUI saves a label,
and deleting it is harmless (it is a report, not the source of truth).

Without labels (or when the layers stay silent) every veto stands — the
pipeline behaves exactly as before.

Installed globally by run.py / app.py before the grouping passes (same
pattern as similarity.set_synonym_families); calibrate.py never installs
it, so calibration measures the raw gate.
"""
from __future__ import annotations

import json
from pathlib import Path

from .similarity import normalize, similarity

# config.yaml -> gate_feedback: {min_labels, soft_threshold, near_precedent}
_DEFAULTS = {
    "min_labels": 3,       # labels on one rule before its weight may act
    "soft_threshold": 0.7,  # share of "merge" labels that softens the rule
    "near_precedent": 0.75,  # lexical similarity to transfer a precedent
}


class GatePrecedents:
    def __init__(self, labels: dict | None, config: dict | None = None,
                 shared_stats: dict | None = None):
        cfg = {**_DEFAULTS, **(config or {})}
        self.min_labels = int(cfg["min_labels"])
        self.soft_threshold = float(cfg["soft_threshold"])
        self.near_sim = float(cfg["near_precedent"])
        # optional external per-rule tally (reason -> [keep_n, merge_n]) to
        # drive the rule-weights layer instead of self.reason_stats. Normally
        # left None: load_gate_precedents already pools ALL products' labels,
        # so self.reason_stats is global. Used only by the GUI weights caption
        # (which builds from one product's labels but wants global counts).
        self.shared_stats = shared_stats
        # (category, norm(a), norm(b)) -> "keep"|"merge"; both orders stored:
        # on a later run the merge direction may flip (votes decide the keep
        # side), and the human verdict is about the PAIR, not the direction
        self.exact: dict[tuple[str, str, str], str] = {}
        self.items: list[dict] = []
        # violated-rule string -> [keep_n, merge_n]
        self.reason_stats: dict[str, list[int]] = {}
        for rec in (labels or {}).values():
            lab = rec.get("label")
            if lab not in ("keep", "merge"):
                continue
            cat = rec.get("category", "")
            a, b = normalize(rec.get("phrase", "")), normalize(rec.get("into", ""))
            if not a or not b:
                continue
            self.exact[(cat, a, b)] = lab
            self.exact[(cat, b, a)] = lab
            self.items.append(rec)
            reason = (rec.get("reason") or "").strip()
            if reason:
                st = self.reason_stats.setdefault(reason, [0, 0])
                st[1 if lab == "merge" else 0] += 1

    def effective_stats(self) -> dict[str, list[int]]:
        """The tally rule_softness runs on: the shared cross-product weights
        when installed, else this product's own labels."""
        return (self.shared_stats if self.shared_stats is not None
                else self.reason_stats)

    def rule_softness(self, reason: str) -> tuple[int, int, bool]:
        """(keep_n, merge_n, is_soft) for one violated-rule string. Counts
        come from the shared cross-product tally when installed."""
        keep_n, merge_n = self.effective_stats().get(reason, (0, 0))
        n = keep_n + merge_n
        soft = n >= self.min_labels and merge_n / n >= self.soft_threshold
        return keep_n, merge_n, soft

    def _near(self, category: str, phrase: str, into: str, reason: str):
        """Best labeled pair with the same rule whose both sides are close
        to the current pair (either alignment). Returns (score, label, rec)
        or None. The min() over the two sides keeps this conservative: ONE
        matching side never transfers a verdict."""
        best = None
        for rec in self.items:
            if rec.get("category") != category:
                continue
            if (rec.get("reason") or "").strip() != reason:
                continue
            pa, pb = rec.get("phrase", ""), rec.get("into", "")
            s = max(min(similarity(phrase, pa), similarity(into, pb)),
                    min(similarity(phrase, pb), similarity(into, pa)))
            if s >= self.near_sim and (best is None or s > best[0]):
                best = (s, rec["label"], rec)
        return best

    def veto_verdict(self, category: str, phrase: str, into: str,
                     reason: str) -> tuple[bool, str]:
        """Should this gate veto stand? -> (blocked, basis). basis names the
        human evidence that decided ('' = no evidence, default block)."""
        exact = self.exact.get((category, normalize(phrase), normalize(into)))
        if exact == "keep":
            return True, "точний прецедент ✓ (ви підтвердили вето)"
        if exact == "merge":
            return False, "точний прецедент ✗ (ви позначили «злити»)"
        near = self._near(category, phrase, into, reason)
        if near is not None:
            s, lab, rec = near
            pair = f"«{rec.get('phrase', '')}» ~ «{rec.get('into', '')}»"
            if lab == "keep":
                return True, f"схожий прецедент ✓ {pair} (схожість {s:.2f})"
            return False, f"схожий прецедент ✗ {pair} (схожість {s:.2f})"
        keep_n, merge_n, soft = self.rule_softness(reason)
        if soft:
            return False, (f"правило ослаблене вагами: {merge_n} з "
                           f"{keep_n + merge_n} вето позначено хибними")
        return True, ""


_ACTIVE: GatePrecedents | None = None

# Shared rule-weight tally lives in the products root (the parent of the
# per-product folders), so every product contributes to and reads the same
# universal-grammar softness decision.
SHARED_WEIGHTS_FILENAME = "gate_rule_weights.json"


def aggregate_all_labels(products_root: Path) -> dict:
    """Merge EVERY product folder's gate_labels.json into one pool, so that
    exact and near precedents (not only rule weights) transfer across
    products. Keys are namespaced by folder name to avoid collisions; the
    records keep their category/phrase/into/reason/label as written."""
    merged: dict = {}
    for lp in sorted(Path(products_root).glob("*/gate_labels.json")):
        try:
            labels = json.loads(lp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stem = lp.parent.name
        for k, rec in labels.items():
            merged[f"{stem}|{k}"] = rec
    return merged


def aggregate_rule_weights(products_root: Path) -> dict[str, list[int]]:
    """Sum per-rule ✓/✗ counts across EVERY product folder's gate_labels.json
    under products_root. reason -> [keep_n, merge_n]. Only these universal
    rule weights cross products; exact/near precedents (lexical) do not."""
    stats: dict[str, list[int]] = {}
    for lp in sorted(Path(products_root).glob("*/gate_labels.json")):
        try:
            labels = json.loads(lp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for rec in labels.values():
            lab = rec.get("label")
            if lab not in ("keep", "merge"):
                continue
            reason = (rec.get("reason") or "").strip()
            if not reason:
                continue
            st = stats.setdefault(reason, [0, 0])
            st[1 if lab == "merge" else 0] += 1
    return stats


def rebuild_shared_weights(products_root: Path) -> dict[str, list[int]]:
    """Recompute the shared rule-weight file from every product's labels and
    persist it to products_root/SHARED_WEIGHTS_FILENAME (best effort — a write
    failure never breaks a run). Returns the aggregate tally."""
    stats = aggregate_rule_weights(products_root)
    p = Path(products_root) / SHARED_WEIGHTS_FILENAME
    try:
        if stats:
            p.write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        elif p.exists():
            p.unlink()
    except OSError:
        pass
    return stats


def set_gate_precedents(labels: dict | None,
                        config: dict | None = None,
                        shared_stats: dict | None = None) -> GatePrecedents:
    global _ACTIVE
    _ACTIVE = GatePrecedents(labels, config, shared_stats)
    return _ACTIVE


def load_gate_precedents(folder: Path,
                         config: dict | None = None) -> GatePrecedents:
    folder = Path(folder)
    root = folder.parent
    # pool labels from ALL products so exact/near precedents AND rule weights
    # are global — a verdict on one product helps every other product.
    merged = aggregate_all_labels(root)
    # persist the human-readable weights report next to the product folders
    # on every load, so CLI runs need no GUI visit first.
    rebuild_shared_weights(root)
    return set_gate_precedents(merged, config)


def veto_verdict(category: str, phrase: str, into: str,
                 reason: str) -> tuple[bool, str]:
    """Module-level entry the grouping passes call; with nothing installed
    every veto stands (pre-feedback behaviour)."""
    if _ACTIVE is None:
        return True, ""
    return _ACTIVE.veto_verdict(category, phrase, into, reason)
