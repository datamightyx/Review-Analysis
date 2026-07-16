"""One-time migration of the legacy flat-file state into SQLite.

For every product folder (default: all of products/*/):
    taxonomy.json    -> <folder>/scoring.db  (taxonomy_groups / canonical_rows
                                              / verbatim_votes / meta)
    .llm_cache.json  -> <folder>/scoring.db  (llm_cache)
And at the project root:
    usage_history.jsonl -> review_scoring.db (usage_history)

The original JSON files are left untouched (delete them yourself once you
trust the migration). Safe to re-run: the taxonomy import is a full
rewrite, the cache import is INSERT OR IGNORE, and the usage history is
skipped if the table already has rows.

Usage:
    python scripts/migrate_to_sqlite.py            # migrate everything
    python scripts/migrate_to_sqlite.py "products/wound closure"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.models import Taxonomy, Group, Canonical
from storage.db_client import product_db, root_db


def load_json_taxonomy(path: Path) -> Taxonomy:
    """Reader for the legacy taxonomy.json format."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tax = Taxonomy()
    tax._next_id = data.get("next_id", 1)
    for g in data.get("groups", []):
        tax.groups[g["id"]] = Group(**g)
    for c in data.get("canonicals", []):
        tax.canonicals[c["id"]] = Canonical(**c)
    return tax


def migrate_folder(folder: Path) -> bool:
    tax_path = folder / "taxonomy.json"
    cache_path = folder / ".llm_cache.json"
    if not tax_path.exists() and not cache_path.exists():
        return False
    db = product_db(folder)
    print(f"{folder.name}: -> {db.path.name}")
    if tax_path.exists():
        tax = load_json_taxonomy(tax_path)
        db.save_taxonomy(tax)
        groups_n, canon_n = db.taxonomy_counts()
        print(f"  taxonomy.json: {groups_n} груп, {canon_n} рядків")
        # sanity: votes must round-trip (COUNT(DISTINCT review_id) == JSON)
        back = db.load_taxonomy()
        for cid, c in tax.canonicals.items():
            for p, v in c.votes.items():
                got = back.canonicals[cid].votes.get(p, 0)
                if got != v:
                    print(f"  (!) голоси розійшлися: {cid}/{p}: "
                          f"json={v} db={got}")
    if cache_path.exists():
        try:
            entries = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  (!) .llm_cache.json не читається, пропущено: {e}")
            entries = {}
        if entries:
            added = db.llm_cache_import(entries)
            print(f"  .llm_cache.json: {len(entries)} записів "
                  f"({added} нових)")
    return True


def migrate_usage(root: Path) -> None:
    log = root / "usage_history.jsonl"
    if not log.exists():
        return
    db = root_db(root)
    if db.usage_count():
        print(f"usage_history: у {db.path.name} вже є записи — пропущено")
        return
    n = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            db.append_usage(json.loads(line))
            n += 1
    print(f"usage_history.jsonl: {n} прогонів -> {db.path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="*",
                    help="product folders (default: all of products/*/)")
    args = ap.parse_args()

    folders = ([Path(f) for f in args.folders] if args.folders
               else sorted(d for d in (ROOT / "products").iterdir()
                           if d.is_dir()))
    migrated = 0
    for folder in folders:
        if not folder.exists():
            sys.exit(f"Немає папки: {folder}")
        migrated += migrate_folder(folder)
    if not args.folders:
        migrate_usage(ROOT)
    print(f"Готово: {migrated} папок мігровано.")


if __name__ == "__main__":
    main()
