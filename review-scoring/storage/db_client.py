"""SQLite persistence layer for the review-scoring pipeline.

Replaces the flat JSON files that used to live in each product folder:
    taxonomy.json      -> taxonomy_groups / canonical_rows / verbatim_votes
    .llm_cache.json    -> llm_cache
    usage_history.jsonl-> usage_history (root-level review_scoring.db)

Design notes
------------
* One `scoring.db` per product folder (taxonomy + votes + LLM cache), one
  `review_scoring.db` at the project root (cross-product usage history).
* WAL journal mode so the Streamlit GUI can read while a pipeline run is
  writing; `check_same_thread=False` + an RLock because Streamlit executes
  the script in worker threads.
* The ONE REVIEW = ONE VOTE invariant is enforced by the storage layer
  itself: vote counts are never stored, they are derived on load with
  `COUNT(DISTINCT review_id)` per (canonical, product).
* Dual placement is supported naturally: the same (product, review_id,
  quote) may appear under two different canonical_ids — the UNIQUE
  constraint includes canonical_id.
* `save_taxonomy` rewrites the taxonomy tables in a single transaction, so
  on re-runs the structure persists (it was loaded first) while votes are
  cleared and rebuilt from the run's actual review ids.

Row model for verbatim_votes: the in-memory Canonical keeps per-product
*parallel* lists of review ids and raw quote variants (they are not paired
per review). They are stored zipped ("" fills the shorter side) so one
table holds both losslessly; vote counting ignores empty review_ids.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

from pipeline.models import Taxonomy, Group, Canonical

PRODUCT_DB_NAME = "scoring.db"
ROOT_DB_NAME = "review_scoring.db"

# review ids synthesized for legacy vote cells that had a count but no
# recorded review ids — they keep COUNT(DISTINCT review_id) correct but are
# filtered back out of Canonical.review_ids on load
_LEGACY_ID_PREFIX = "__legacy_"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS taxonomy_groups (
    group_id       TEXT PRIMARY KEY,
    category       TEXT NOT NULL,
    name           TEXT NOT NULL,
    usage_category TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS canonical_rows (
    canonical_id TEXT PRIMARY KEY,
    group_id     TEXT NOT NULL REFERENCES taxonomy_groups(group_id)
                 ON DELETE CASCADE,
    text         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verbatim_votes (
    vote_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id TEXT NOT NULL REFERENCES canonical_rows(canonical_id)
                 ON DELETE CASCADE,
    product      TEXT NOT NULL,
    review_id    TEXT NOT NULL DEFAULT '',
    quote        TEXT NOT NULL DEFAULT '',
    UNIQUE (canonical_id, product, review_id, quote)
);
CREATE INDEX IF NOT EXISTS idx_votes_canonical
    ON verbatim_votes (canonical_id, product);
CREATE TABLE IF NOT EXISTS quote_sources (
    canonical_id TEXT NOT NULL REFERENCES canonical_rows(canonical_id)
                 ON DELETE CASCADE,
    product      TEXT NOT NULL,
    quote        TEXT NOT NULL,
    review_id    TEXT NOT NULL,
    UNIQUE (canonical_id, product, quote, review_id)
);
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key     TEXT PRIMARY KEY,
    provider      TEXT NOT NULL DEFAULT '',
    model         TEXT NOT NULL DEFAULT '',
    effort        TEXT NOT NULL DEFAULT '',
    response_text TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_history (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT NOT NULL,
    date               TEXT NOT NULL,
    product_line       TEXT NOT NULL,
    provider           TEXT NOT NULL DEFAULT '',
    extract_model      TEXT NOT NULL DEFAULT '',
    group_model        TEXT NOT NULL DEFAULT '',
    reviews            INTEGER NOT NULL DEFAULT 0,
    phrases            INTEGER NOT NULL DEFAULT 0,
    groups             INTEGER NOT NULL DEFAULT 0,
    calls              INTEGER NOT NULL DEFAULT 0,
    cache_hits         INTEGER NOT NULL DEFAULT 0,
    input_tokens       INTEGER NOT NULL DEFAULT 0,
    output_tokens      INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd           REAL NOT NULL DEFAULT 0.0,
    cost_known         INTEGER NOT NULL DEFAULT 1
);
"""

USAGE_COLUMNS = [
    "timestamp", "date", "product_line", "provider",
    "extract_model", "group_model", "reviews", "phrases", "groups",
    "calls", "cache_hits", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "cost_usd", "cost_known",
]


class DB:
    """One SQLite file; safe to share across pipeline threads / Streamlit
    reruns (single connection, RLock-guarded, WAL journal)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection (needed on Windows before the
        .db file can be deleted; the app itself never calls this)."""
        with self._lock:
            self._conn.close()

    def checkpoint(self) -> None:
        """Flush the WAL into the main .db file so the file on disk is a
        complete, self-contained snapshot — needed before uploading it
        anywhere (R2, backups) since -wal/-shm siblings aren't shipped."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(FULL)")

    # ---------- taxonomy ----------

    def save_taxonomy(self, tax: Taxonomy) -> None:
        """Full transactional rewrite of the taxonomy tables from the
        in-memory Taxonomy. Row insertion order preserves the taxonomy's
        dict order, so a load round-trips to the same iteration order
        (Excel tie-breaking stays byte-identical)."""
        vote_rows: list[tuple[str, str, str, str]] = []
        source_rows: list[tuple[str, str, str, str]] = []
        for cn in tax.canonicals.values():
            products = list(dict.fromkeys(
                [*cn.votes, *cn.quotes, *cn.review_ids]))
            for product in products:
                ids = list(dict.fromkeys(cn.review_ids.get(product, [])))
                if not ids:  # legacy cell: a count with no recorded ids
                    ids = [f"{_LEGACY_ID_PREFIX}{i + 1}"
                           for i in range(cn.votes.get(product, 0))]
                quotes = list(dict.fromkeys(cn.quotes.get(product, [])))
                for rid, quote in zip_longest(ids, quotes, fillvalue=""):
                    vote_rows.append((cn.id, product, rid, quote))
            for product, smap in cn.quote_sources.items():
                for quote, rids in smap.items():
                    for rid in rids:
                        source_rows.append((cn.id, product, quote, rid))
        with self._lock, self._conn:
            c = self._conn
            c.execute("DELETE FROM verbatim_votes")
            c.execute("DELETE FROM quote_sources")
            c.execute("DELETE FROM canonical_rows")
            c.execute("DELETE FROM taxonomy_groups")
            c.execute("INSERT OR REPLACE INTO meta VALUES ('next_id', ?)",
                      (str(tax._next_id),))
            c.executemany(
                "INSERT INTO taxonomy_groups VALUES (?, ?, ?, ?)",
                [(g.id, g.category, g.name, g.usage_category)
                 for g in tax.groups.values()])
            c.executemany(
                "INSERT INTO canonical_rows VALUES (?, ?, ?)",
                [(cn.id, cn.group_id, cn.text)
                 for cn in tax.canonicals.values()])
            c.executemany(
                "INSERT OR IGNORE INTO verbatim_votes "
                "(canonical_id, product, review_id, quote) VALUES (?, ?, ?, ?)",
                vote_rows)
            c.executemany(
                "INSERT OR IGNORE INTO quote_sources "
                "(canonical_id, product, quote, review_id) VALUES (?, ?, ?, ?)",
                source_rows)
        self.checkpoint()

    def load_taxonomy(self) -> Taxonomy:
        """Rebuild the in-memory Taxonomy. Vote counts come from a single
        aggregate query — COUNT(DISTINCT review_id) per (canonical, product)
        — so the one-review-one-vote invariant holds by construction."""
        tax = Taxonomy()
        with self._lock:
            c = self._conn
            row = c.execute(
                "SELECT value FROM meta WHERE key = 'next_id'").fetchone()
            tax._next_id = int(row[0]) if row else 1
            for gid, cat, name, ucat in c.execute(
                    "SELECT group_id, category, name, usage_category "
                    "FROM taxonomy_groups ORDER BY rowid"):
                tax.groups[gid] = Group(id=gid, category=cat, name=name,
                                        usage_category=ucat)
            for cid, gid, text in c.execute(
                    "SELECT canonical_id, group_id, text "
                    "FROM canonical_rows ORDER BY rowid"):
                tax.canonicals[cid] = Canonical(id=cid, text=text,
                                                group_id=gid)
            for cid, product, rid, quote in c.execute(
                    "SELECT v.canonical_id, v.product, v.review_id, v.quote "
                    "FROM verbatim_votes v "
                    "JOIN canonical_rows r ON r.canonical_id = v.canonical_id "
                    "ORDER BY v.vote_id"):
                cn = tax.canonicals.get(cid)
                if cn is None:
                    continue
                cn.votes.setdefault(product, 0)  # fixes product order
                if quote:
                    qs = cn.quotes.setdefault(product, [])
                    if quote not in qs:
                        qs.append(quote)
                if rid and not rid.startswith(_LEGACY_ID_PREFIX):
                    ids = cn.review_ids.setdefault(product, [])
                    if rid not in ids:
                        ids.append(rid)
            for cid, product, n in c.execute(
                    "SELECT canonical_id, product, COUNT(DISTINCT review_id) "
                    "FROM verbatim_votes WHERE review_id != '' "
                    "GROUP BY canonical_id, product"):
                if cid in tax.canonicals:
                    tax.canonicals[cid].votes[product] = n
            for cid, product, quote, rid in c.execute(
                    "SELECT canonical_id, product, quote, review_id "
                    "FROM quote_sources ORDER BY rowid"):
                cn = tax.canonicals.get(cid)
                if cn is None:
                    continue
                smap = cn.quote_sources.setdefault(product, {})
                lst = smap.setdefault(quote, [])
                if rid not in lst:
                    lst.append(rid)
        return tax

    def taxonomy_counts(self) -> tuple[int, int]:
        """(groups, canonicals) — cheap header stats without loading."""
        with self._lock:
            row = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM taxonomy_groups), "
                "       (SELECT COUNT(*) FROM canonical_rows)").fetchone()
        return row[0], row[1]

    def has_taxonomy(self) -> bool:
        return self.taxonomy_counts()[0] > 0

    # ---------- LLM cache ----------

    def llm_cache_get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response_text FROM llm_cache WHERE cache_key = ?",
                (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def llm_cache_put(self, key: str, provider: str, model: str,
                      effort: str, result: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?)",
                (key, provider, model, effort,
                 json.dumps(result, ensure_ascii=False),
                 datetime.now().isoformat(timespec="seconds")))

    def llm_cache_import(self, entries: dict[str, dict]) -> int:
        """Bulk import of legacy .llm_cache.json entries ({key: result}).
        Provider/model/effort are unknown for old entries — stored empty.
        Returns how many rows were actually inserted."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._conn:
            before = self._conn.execute(
                "SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            self._conn.executemany(
                "INSERT OR IGNORE INTO llm_cache VALUES (?, '', '', '', ?, ?)",
                [(k, json.dumps(v, ensure_ascii=False), now)
                 for k, v in entries.items()])
            after = self._conn.execute(
                "SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        return after - before

    # ---------- usage history (root-level DB) ----------

    def append_usage(self, record: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO usage_history ({', '.join(USAGE_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(USAGE_COLUMNS))})",
                [int(record[c]) if c == "cost_known" else record.get(c)
                 for c in USAGE_COLUMNS])
        self.checkpoint()

    def load_usage(self) -> list[dict]:
        """All logged pipeline runs, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(USAGE_COLUMNS)} FROM usage_history "
                "ORDER BY id").fetchall()
        out = []
        for r in rows:
            rec = dict(zip(USAGE_COLUMNS, r))
            rec["cost_known"] = bool(rec["cost_known"])
            out.append(rec)
        return out

    def usage_count(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM usage_history").fetchone()[0]


# ---------- connection registry ----------
# one DB object (one connection) per file, shared across the process — the
# five LLM instances of a run and every Streamlit rerun reuse it

_open: dict[str, DB] = {}
_open_lock = threading.Lock()


def open_db(path: Path | str) -> DB:
    key = str(Path(path).resolve())
    with _open_lock:
        db = _open.get(key)
        if db is None:
            db = _open[key] = DB(path)
        return db


def product_db(folder: Path | str) -> DB:
    """The per-product-folder database (taxonomy, votes, LLM cache)."""
    return open_db(Path(folder) / PRODUCT_DB_NAME)


def close_product_db(folder: Path | str) -> None:
    """Close and evict a product folder's DB connection from the registry.
    Required before deleting the folder — Windows keeps an open sqlite file
    locked, and a stale registry entry would hand out a closed connection
    if a folder with the same name were recreated later in the same
    process. The pop AND the close happen under the SAME `_open_lock` hold
    (not released in between) so another thread's open_db() for this same
    path — which also needs `_open_lock` — cannot race in and reopen the
    file while it's being closed for deletion."""
    key = str((Path(folder) / PRODUCT_DB_NAME).resolve())
    with _open_lock:
        db = _open.pop(key, None)
        if db is not None:
            db.close()


def root_db(root: Path | str) -> DB:
    """The project-root database (cross-product usage history)."""
    return open_db(Path(root) / ROOT_DB_NAME)
