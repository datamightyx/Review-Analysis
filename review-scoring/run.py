"""Review scoring pipeline.

Usage:
    python run.py <product_folder> [options]

The folder must contain the Amazon review PDF exports. On the first run a
products.yaml skeleton is generated — fill in the short product names and
links, then run again.

Options:
    --max-reviews N    max reviews per product (default: no limit, whole PDF)
    --cutoff DATE      only reviews on/after YYYY-MM-DD
    --limit N          take only first N reviews per product (quick test)
    --fresh            rebuild the taxonomy from scratch
    --excel-only       just regenerate the .xlsx from the taxonomy stored
                       in scoring.db (use after editing overrides.json)
    --model NAME       override the model from config.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.pdf_parser import parse_pdf, filter_reviews
from pipeline.extract import (extract_phrases, validate_verbatim,
                              dedupe_overlapping)
from pipeline.grouping import (group_phrases, apply_overrides,
                               consolidate_taxonomy, reassign_phrases,
                               consolidate_rows, merge_sibling_rows,
                               reconcile_votes)
from pipeline.excel_writer import save_workbook
from pipeline.similarity import set_synonym_families
from pipeline.precedents import load_gate_precedents
from pipeline.llm import LLM, set_max_concurrency
from pipeline.models import Review, ExtractedPhrase, Taxonomy, product_key
from pipeline import domain as domain_mod
from storage.db_client import product_db

ROOT = Path(__file__).parent




def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return {}


def ensure_products_yaml(folder: Path) -> dict[str, dict]:
    py = folder / "products.yaml"
    pdfs = sorted(folder.glob("*.pdf"))
    if not py.exists():
        # default the name to the product key (sentiment suffix stripped) so a
        # product's positive & negative PDFs share ONE column out of the box
        skeleton = {
            p.stem: {"name": product_key(p.stem), "link": ""} for p in pdfs
        }
        py.write_text(
            yaml.safe_dump(skeleton, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"Створено {py}")
        print("Заповніть короткі назви продуктів (name) і посилання (link), "
              "потім запустіть ще раз.")
        sys.exit(0)
    mapping = yaml.safe_load(py.read_text(encoding="utf-8")) or {}
    missing = [p.stem for p in pdfs if p.stem not in mapping]
    if missing:
        print("У products.yaml немає записів для:", *missing, sep="\n  ")
        sys.exit(1)
    return mapping


def main() -> None:
    # Windows console is cp1251 — a review quote with any char outside it
    # (accented letters, curly quotes) would crash every print of merge
    # actions; replace unencodable chars instead of dying
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--max-reviews", type=int, default=None)
    ap.add_argument("--cutoff", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--excel-only", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--no-consolidate", action="store_true",
                    help="пропустити завершальний пас консолідації таксономії")
    ap.add_argument("--no-reassign", action="store_true",
                    help="пропустити фінальний пас перепризначення фраз")
    ap.add_argument("--no-row-merge", action="store_true",
                    help="пропустити LLM-пас злиття рядків-синонімів")
    ap.add_argument("--rows-only", action="store_true",
                    help="лише злити рядки-синоніми в наявній таксономії "
                         "(без екстракції/групування) і перегенерувати Excel")
    ap.add_argument("--init-domain", action="store_true",
                    help="створити domain.yaml (профіль категорій та аркушів) "
                         "у папці продукту й вийти — далі відредагуйте його "
                         "під свою групу товарів")
    args = ap.parse_args()

    if args.init_domain:
        folder = Path(args.folder)
        folder.mkdir(parents=True, exist_ok=True)
        dp = folder / "domain.yaml"
        if dp.exists():
            sys.exit(f"{dp} вже існує — не перезаписую.")
        domain_mod.write_default_yaml(dp)
        print(f"Створено {dp}. Відредагуйте категорії/аркуші під свою "
              "групу товарів і запустіть звичайний прогін.")
        return

    cfg = load_config()
    # cap on concurrent LLM calls (asyncio semaphore shared by all passes)
    set_max_concurrency(cfg.get("max_concurrent_requests", 10))
    # optional per-config deterministic synonym families (see config.yaml);
    # must be installed before any grouping/consolidation pass
    set_synonym_families(cfg.get("synonym_families"))
    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"Немає папки: {folder}")
    # per-product domain profile (categories + Excel layout). Absent =>
    # the built-in default (the original five categories). Must be installed
    # before any extraction/grouping/excel pass.
    try:
        domain_mod.set_active_domain(domain_mod.load_domain(folder))
    except (ValueError, KeyError) as e:
        sys.exit(f"Помилка в domain.yaml: {e}")
    # human ✓/✗ verdicts from the review tab feed back into the merge gate
    # (exact/near precedents + per-rule weights) — must be installed before
    # any pass that calls merge_blocked
    load_gate_precedents(folder, cfg.get("gate_feedback"))

    mapping = ensure_products_yaml(folder)
    products = {v["name"]: v.get("link", "") for v in mapping.values()}

    db = product_db(folder)
    out_name = f"{folder.name} - {date.today().strftime('%d.%m.%Y')}.xlsx"
    out_path = folder / out_name

    if args.excel_only:
        tax = db.load_taxonomy()
        apply_overrides(tax, folder / "overrides.json")
        written = save_workbook(tax, products, out_path)
        print(f"Записано {written}")
        return

    if args.rows_only:
        tax = db.load_taxonomy()
        llm_rows = LLM(
            model=args.model or cfg.get("consolidate_model")
            or cfg.get("model"),
            cache=db,
            effort=cfg.get("consolidate_effort") or cfg.get("effort", "medium"),
            provider=cfg.get("provider", "anthropic"),
        )
        audit: list[dict] = []
        actions = consolidate_rows(tax, audit)
        actions += merge_sibling_rows(tax, llm_rows, audit)
        for a in actions:
            print(f"  {a}")
        print(f"Злито рядків: {len(actions)}")
        db.save_taxonomy(tax)
        rq_path = folder / "review_queue.json"
        old_rq = (json.loads(rq_path.read_text(encoding="utf-8"))
                  if rq_path.exists() else [])
        rq_path.write_text(json.dumps(old_rq + audit, ensure_ascii=False,
                                      indent=1), encoding="utf-8")
        apply_overrides(tax, folder / "overrides.json")
        written = save_workbook(tax, products, out_path)
        print(f"  {llm_rows.usage_report()}")
        print(f"Записано {written}")
        return

    max_reviews = args.max_reviews or cfg.get("max_reviews")  # None = весь PDF
    cutoff = None
    if args.cutoff or cfg.get("cutoff"):
        cutoff = datetime.fromisoformat(str(args.cutoff or cfg["cutoff"])).date()

    # ---- 1. parse PDFs ----
    reviews_path = folder / "reviews.json"
    all_reviews: list[Review] = []
    parse_warnings: list[dict] = []
    for pdf in sorted(folder.glob("*.pdf")):
        product = mapping[pdf.stem]["name"]
        pstats: dict = {}
        rs = parse_pdf(pdf, product, stats=pstats)
        if pstats.get("expected", 0) > pstats.get("parsed", 0):
            print(f"  (!) {pdf.name}: розпізнано {pstats['parsed']} з "
                  f"{pstats['expected']} відгуків — втрачені фрагменти "
                  f"у parse_warnings.json")
            parse_warnings.append({"product": product, **pstats})
        rs = filter_reviews(rs, max_reviews=max_reviews, cutoff=cutoff)
        if args.limit:
            rs = rs[:args.limit]
        print(f"  {product}: {len(rs)} відгуків")
        all_reviews.extend(rs)
    pw_path = folder / "parse_warnings.json"
    if parse_warnings:
        pw_path.write_text(json.dumps(parse_warnings, ensure_ascii=False,
                                      indent=1), encoding="utf-8")
    elif pw_path.exists():
        pw_path.unlink()   # stale warnings from an older run
    reviews_path.write_text(
        json.dumps([dataclasses.asdict(r) for r in all_reviews],
                   ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"Разом: {len(all_reviews)} відгуків")

    # ---- 2. LLM setup ----
    # extraction is a simpler "follow the rules" task than grouping, so it
    # gets its own (cheaper/faster) model + effort by default; --model
    # overrides both at once for quick whole-pipeline tests.
    provider = cfg.get("provider", "anthropic")
    llm_group = LLM(
        model=args.model or cfg.get("model"),
        cache=db,
        effort=cfg.get("effort", "medium"),
        provider=provider,
    )
    llm_extract = LLM(
        model=args.model or cfg.get("extract_model") or cfg.get("model"),
        cache=db,
        effort=cfg.get("extract_effort") or cfg.get("effort", "medium"),
        provider=provider,
    )
    # consolidation is a handful of whole-taxonomy calls per run, so a
    # stronger model/effort there costs almost nothing
    llm_consolidate = LLM(
        model=args.model or cfg.get("consolidate_model") or cfg.get("model"),
        cache=db,
        effort=cfg.get("consolidate_effort") or cfg.get("effort", "medium"),
        provider=provider,
    )
    # the double-check ("is this new group a duplicate theme?") is a narrow
    # lexical decision — it gets its own (lower) effort so the extra per-batch
    # call stays cheap; same model as grouping.
    llm_verify = LLM(
        model=args.model or cfg.get("model"),
        cache=db,
        effort=cfg.get("verify_effort") or cfg.get("effort", "medium"),
        provider=provider,
    )
    # reassignment replays the whole corpus against a FINAL (fixed) taxonomy —
    # a matching task, not open-ended taxonomy building. It runs at its own
    # (lower) effort by default: reasoning output tokens are the dominant cost
    # and this pass needs little of it.
    llm_reassign = LLM(
        model=args.model or cfg.get("reassign_model") or cfg.get("model"),
        cache=db,
        effort=cfg.get("reassign_effort") or cfg.get("effort", "medium"),
        provider=provider,
    )

    # ---- 3. extract phrases ----
    def eprog(done, total):
        print(f"  екстракція: {done}/{total} відгуків", end="\r")
    phrases = extract_phrases(all_reviews, llm_extract,
                              batch_size=cfg.get("extract_batch", 8),
                              progress=eprog)
    print(f"\nВитягнуто фраз: {len(phrases)}")

    # verbatim check: exact quotes pass, close paraphrases are repaired to
    # the actual review text, the rest get one LLM repair attempt; still-
    # unverified quotes are dropped and logged to verbatim_warnings.json
    vv = validate_verbatim(phrases, all_reviews, llm=llm_extract)
    if vv["repaired"]:
        print(f"  відновлено дослівність: {len(vv['repaired'])} цитат")
    if vv["unverified"]:
        (folder / "verbatim_warnings.json").write_text(
            json.dumps(vv["unverified"], ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"  (!) вилучено вигадані цитати: {len(vv['unverified'])} "
              f"(див. verbatim_warnings.json)")

    # collapse intra-review overlaps (a short clause + the longer sentence that
    # quotes it) so one review is never counted twice; enumerated second
    # aspects ("...or facial cuts") are split off into their own phrase
    phrases, dd = dedupe_overlapping(phrases)
    if dd["dropped"] or dd["split"]:
        print(f"  усунено перекриття фраз: {len(dd['dropped'])} прибрано, "
              f"{len(dd['split'])} розділено")

    (folder / "phrases.json").write_text(
        json.dumps([dataclasses.asdict(p) for p in phrases],
                   ensure_ascii=False, indent=1),
        encoding="utf-8")

    # ---- 4. group ----
    # structure persists across runs (loaded from the DB); votes are cleared
    # and rebuilt for the run's products when the taxonomy is saved back
    tax = Taxonomy() if args.fresh else db.load_taxonomy()

    audit: list[dict] = []

    def gprog(cat, done, total):
        print(f"  групування [{cat}]: {done}/{total} фраз      ", end="\r")
    tax = group_phrases(phrases, tax, llm_group,
                        batch_size=cfg.get("group_batch", 25),
                        progress=gprog,
                        double_check=cfg.get("double_check", True),
                        audit=audit,
                        verify_llm=llm_verify)
    print()

    # ---- 4b. consolidation pass over the whole taxonomy ----
    if not args.no_consolidate and cfg.get("consolidate", True):
        actions = consolidate_taxonomy(tax, llm_consolidate)
        if actions:
            print("Консолідація таксономії:")
            for a in actions:
                print(f"  {a}")
            audit.extend({"type": "consolidate", "category": "",
                          "action": a} for a in actions)

    # ---- 4c. reassignment: replay the corpus against the final taxonomy ----
    if not args.no_reassign and cfg.get("reassign", True):
        def rprog(cat, done, total):
            print(f"  перепризначення [{cat}]: {done}/{total} фраз      ",
                  end="\r")
        r_actions = reassign_phrases(phrases, tax, llm_reassign,
                                     batch_size=cfg.get("group_batch", 25),
                                     progress=rprog, audit=audit)
        print()
        for a in r_actions:
            print(f"  {a}")

    # ---- 4d. LLM row-merge: sibling rows saying the same thing ----
    if not args.no_row_merge and cfg.get("row_merge", True):
        m_actions = merge_sibling_rows(tax, llm_consolidate, audit)
        if m_actions:
            print("Злиття рядків-синонімів:")
            for a in m_actions:
                print(f"  {a}")

    # groups that ended up as a single low-vote row deserve a human look
    for g in tax.groups.values():
        cans = g.canonicals(tax)
        if len(cans) == 1 and cans[0].total <= 2:
            audit.append({"type": "singleton_group", "category": g.category,
                          "group": g.name, "row": cans[0].text,
                          "votes": cans[0].total})

    reconcile_votes(tax)   # persist the one-review-one-vote invariant
    db.save_taxonomy(tax)
    apply_overrides(tax, folder / "overrides.json")

    # ---- 4e. review queue for a human ----
    seen: set[str] = set()
    audit = [a for a in audit
             if (key := json.dumps(a, ensure_ascii=False, sort_keys=True))
             not in seen and not seen.add(key)]
    (folder / "review_queue.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    if audit:
        print(f"  (!) місць для людської перевірки: {len(audit)} "
              f"(review_queue.json / вкладка «Перевірити» в GUI)")

    # ---- 5. excel ----
    written = save_workbook(tax, products, out_path)
    all_llms = [llm_extract, llm_group, llm_verify, llm_reassign, llm_consolidate]
    print(f"  екстракція: {llm_extract.usage_report()}")
    print(f"  групування: {llm_group.usage_report()}")
    if llm_verify.calls:
        print(f"  перевірка груп: {llm_verify.usage_report()}")
    if llm_reassign.calls:
        print(f"  перепризначення: {llm_reassign.usage_report()}")
    print(f"  консолідація: {llm_consolidate.usage_report()}")
    print(LLM.combined_usage_report(all_llms))
    print(f"Готово: {written}")


if __name__ == "__main__":
    main()
