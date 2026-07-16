"""Calibrate the grouping judge against the reference workbook.

Reads the human-made reference (Styptic powder xlsx), takes the review
phrases with their gold USP groups, runs our grouping pipeline on them from
scratch, and reports pairwise precision/recall of "same group" decisions,
plus any over-merges at the canonical level (reference rows are already
human-deduplicated, so two different rows of one product merged into one
canonical = over-merge).

Usage:
    python calibrate.py "..\\Styptic powder - 25.05.2026.xlsx" [--limit 150]
                        [--sheet Positive] [--model claude-opus-4-8]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.grouping import (group_phrases, normalize, consolidate_taxonomy,
                               reassign_phrases, merge_sibling_rows)
from pipeline.llm import LLM
from pipeline.models import ExtractedPhrase, Taxonomy


def read_gold(xlsx: Path, sheet: str) -> list[dict]:
    """Rows: {group, text, product, votes} from a Positive/Negative sheet."""
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb[sheet]
    rows = []
    current_group = None
    for r in ws.iter_rows(min_row=2, max_col=4, values_only=True):
        group, text, product, votes = r
        if group:
            current_group = str(group).strip()
        if not text or not current_group:
            continue
        rows.append({
            "group": current_group,
            "text": str(text).strip(),
            "product": str(product).strip() if product else "?",
            "votes": int(votes) if isinstance(votes, (int, float)) else 1,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--sheet", default="Positive")
    ap.add_argument("--per-group", type=int, default=10,
                    help="рядків з кожної еталонної групи (стратифікована вибірка)")
    ap.add_argument("--limit", type=int, default=None,
                    help="загальний ліміт рядків (після стратифікації)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    gold_rows = read_gold(Path(args.xlsx), args.sheet)
    print(f"Еталон: {len(gold_rows)} рядків, "
          f"{len({r['group'] for r in gold_rows})} груп")

    # stratified sample: top-N rows from every gold group, so the metric
    # covers the whole taxonomy, not just the first (largest) group
    taken = defaultdict(int)
    rows = []
    for r in gold_rows:
        if taken[r["group"]] < args.per_group:
            taken[r["group"]] += 1
            rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Вибірка: {len(rows)} рядків з {len(taken)} груп")
    category = "positive" if "positive" in args.sheet.lower() else "negative"

    # replicate vote counts so frequency-based ordering works as in production
    phrases = []
    for i, r in enumerate(rows):
        for _ in range(max(1, min(r["votes"], 10))):
            phrases.append(ExtractedPhrase(
                quote=r["text"], category=category,
                product=r["product"], review_id=f"gold:{i}",
            ))

    import yaml
    cfg_path = Path(__file__).parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    provider = cfg.get("provider", "anthropic")
    from pipeline.llm import set_max_concurrency
    set_max_concurrency(cfg.get("max_concurrent_requests", 10))
    from storage.db_client import root_db
    cache = root_db(Path(__file__).parent)
    # mirror run.py's tiered LLM setup (same cache DB, so a rerun and even
    # cross-pass calls with identical prompts hit cache) so this measures the
    # SAME pipeline the user actually runs, not just the first grouping pass.
    llm = LLM(model=args.model or cfg.get("model"),
              cache=cache,
              effort=cfg.get("effort", "medium"),
              provider=provider)
    llm_consolidate = LLM(
        model=args.model or cfg.get("consolidate_model") or cfg.get("model"),
        cache=cache,
        effort=cfg.get("consolidate_effort") or cfg.get("effort", "medium"),
        provider=provider)
    llm_reassign = LLM(
        model=args.model or cfg.get("reassign_model") or cfg.get("model"),
        cache=cache,
        effort=cfg.get("reassign_effort") or cfg.get("effort", "medium"),
        provider=provider)

    tax = Taxonomy()

    def prog(cat, done, total):
        print(f"  групування: {done}/{total}", end="\r")
    group_phrases(phrases, tax, llm, progress=prog,
                  double_check=cfg.get("double_check", True))
    print()

    if cfg.get("consolidate", True):
        actions = consolidate_taxonomy(tax, llm_consolidate)
        if actions:
            print("Консолідація:")
            for a in actions:
                print(f"  {a}")

    if cfg.get("reassign", True):
        def rprog(cat, done, total):
            print(f"  перепризначення: {done}/{total}", end="\r")
        r_actions = reassign_phrases(phrases, tax, llm_reassign, progress=rprog)
        print()
        if r_actions:
            print("Перепризначення:")
            for a in r_actions:
                print(f"  {a}")

    if cfg.get("row_merge", True):
        m_actions = merge_sibling_rows(tax, llm_consolidate)
        if m_actions:
            print("Злиття рядків-синонімів:")
            for a in m_actions:
                print(f"  {a}")

    # predicted group per normalized phrase text — read straight from the
    # FINAL taxonomy state (canon.quotes), not a first-pass log: reassign
    # rebuilds votes/quotes from scratch and row_merge can change which
    # canonical a text lives under, so a first-pass log would be stale.
    pred_group: dict[str, str] = {}
    canon_members: dict[str, list[str]] = defaultdict(list)
    for cid, canon in tax.canonicals.items():
        for texts in canon.quotes.values():
            for text in texts:
                pred_group[normalize(text)] = canon.group_id
                canon_members[cid].append(text)

    # unique gold items present in prediction
    items = []
    seen = set()
    for r in rows:
        key = normalize(r["text"])
        if key in seen or key not in pred_group:
            continue
        seen.add(key)
        items.append((key, r["group"], pred_group[key], r["text"], r["product"]))

    tp = fp = fn = 0
    fp_examples, fn_examples = [], []
    for (k1, g1, p1, t1, _), (k2, g2, p2, t2, _) in combinations(items, 2):
        same_gold = g1 == g2
        same_pred = p1 == p2
        if same_pred and same_gold:
            tp += 1
        elif same_pred and not same_gold:
            fp += 1
            if len(fp_examples) < 12:
                fp_examples.append((t1, g1, t2, g2))
        elif same_gold and not same_pred:
            fn += 1
            if len(fn_examples) < 12:
                fn_examples.append((t1, t2, g1))

    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    print(f"\nПопарні метрики (same group): "
          f"precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print(f"Груп в еталоні (на вибірці): {len({g for _, g, *_ in items})}, "
          f"у передбаченні: {len({p for _, _, p, *_ in items})}")

    if fp_examples:
        print("\nПомилково в одній групі (приклади):")
        for t1, g1, t2, g2 in fp_examples:
            print(f'  "{t1}" [{g1}]  +  "{t2}" [{g2}]')
    if fn_examples:
        print("\nПомилково розділені (приклади):")
        for t1, t2, g in fn_examples:
            print(f'  "{t1}"  |  "{t2}"   (еталонна група: {g})')

    # canonical-level over-merges: reference rows are human-deduplicated
    # WITHIN a product, so merging two different texts of the SAME product
    # is suspicious; cross-product merges of identical wording are fine.
    products_by_key: dict[str, set] = defaultdict(set)
    text_by_key = {}
    for r in rows:
        k = normalize(r["text"])
        products_by_key[k].add(r["product"])
        text_by_key[k] = r["text"]
    over = []
    for cid, members in canon_members.items():
        uniq = sorted({normalize(m) for m in members})
        if len(uniq) < 2:
            continue
        prod_seen = defaultdict(list)
        for k in uniq:
            for p in products_by_key.get(k, ()):
                prod_seen[p].append(k)
        if any(len(ks) > 1 for ks in prod_seen.values()):
            over.append(sorted({text_by_key[k] for k in uniq if k in text_by_key}))
    print(f"\nПідозрілі злиття (різні еталонні рядки одного продукту): {len(over)}")
    for texts in over[:15]:
        print("  MERGE:", " | ".join(texts))

    total_calls = llm.calls + llm_consolidate.calls + llm_reassign.calls
    total_hits = llm.cache_hits + llm_consolidate.cache_hits + llm_reassign.cache_hits
    print(f"\nLLM: {total_calls} викликів, {total_hits} з кешу")


if __name__ == "__main__":
    main()
