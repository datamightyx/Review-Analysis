"""Streamlit GUI for the review-scoring pipeline.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import html
import json
import os
import shutil
import sys
import zlib
from datetime import date, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# pipeline/llm.py reads provider keys straight from os.environ; locally
# that's whatever's already set in the shell / .env, but Streamlit Cloud
# secrets are only reachable via st.secrets — mirror them into the
# environment here (once) so the pipeline code doesn't need to change.
for _key in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
    if _key not in os.environ:
        try:
            os.environ[_key] = st.secrets[_key]
        except Exception:
            pass

from pipeline.pdf_parser import parse_pdf, filter_reviews
from pipeline.extract import (extract_phrases, validate_verbatim,
                              dedupe_overlapping)
from pipeline.grouping import (group_phrases, apply_overrides, normalize,
                               reassign_phrases, reconcile_votes,
                               consolidate_taxonomy, merge_sibling_rows,
                               split_groups, prune_empty)
from pipeline import manual_edits as me
from components import taxonomy_board
from pipeline.excel_writer import save_workbook
from pipeline.quality import workbook_warnings
from pipeline.llm import LLM, set_max_concurrency, ANTHROPIC_PRICING
from pipeline.models import Taxonomy, product_key
from pipeline.precedents import (GatePrecedents, aggregate_rule_weights,
                                 load_gate_precedents, rebuild_shared_weights)
from pipeline import domain as domain_mod
from storage.db_client import (product_db, root_db, close_product_db,
                               PRODUCT_DB_NAME, ROOT_DB_NAME)
from storage import r2_sync

# known models per provider, for the "Моделі за кроками" selects on the Run
# tab — the anthropic list comes straight from the pricing table (llm.py) so
# it can't drift out of sync; openrouter slugs are curated since OpenRouter's
# catalogue is much larger than what this pipeline is tuned for.
MODEL_CHOICES = {
    "anthropic": sorted(ANTHROPIC_PRICING.keys()),
    "openrouter": [
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-oss-120b:free",
    ],
}
MODEL_FROM_CONFIG = "— з config.yaml —"
MODEL_CUSTOM = "Інша (вписати)…"

CAT_LABELS = {
    "positive": "✅ Positive",
    "negative": "❌ Negative",
    "usage": "🐾 Usage",
    "improvement": "🔧 Improvement",
    "who_recommended": "👍 Who recommended",
}


def cat_label(cat: str) -> str:
    """Display label for a category — falls back to the raw id so a custom
    domain profile with new categories still renders."""
    return CAT_LABELS.get(cat, f"🏷 {cat}")

ACCENT = "#2a78d6"  # single-hue accent, used for both dark & light themes

st.set_page_config(page_title="Review Scoring", page_icon="📊", layout="wide")

# Streamlit doesn't expose theme colors as CSS custom properties in this
# version (--primary-color / --secondary-background-color never resolve),
# so panel/border/shadow tints are derived from `currentColor` via
# color-mix() instead — that tracks whichever theme (light or dark) is
# actually active without needing to know which one it is.
st.markdown("""
<style>
:root {
    --acc: #2a78d6;
    --panel: color-mix(in srgb, currentColor 6%, transparent);
    --panel-hover: color-mix(in srgb, currentColor 10%, transparent);
    --border-soft: color-mix(in srgb, currentColor 14%, transparent);
    --border-strong: color-mix(in srgb, currentColor 30%, transparent);
    --shadow-1: color-mix(in srgb, currentColor 8%, transparent);
    --shadow-2: color-mix(in srgb, currentColor 14%, transparent);
}

/* ---- global feel ---- */
.block-container { padding-top: 1.6rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.01em; }

/* ---- bordered containers -> soft cards ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    transition: box-shadow .15s ease, border-color .15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div) {
    box-shadow: 0 1px 2px var(--shadow-1);
}

/* ---- metrics as compact stat chips ---- */
div[data-testid="stMetric"] {
    background: var(--panel);
    border-radius: 12px;
    padding: 0.7rem 1rem 0.55rem;
    border: 1px solid var(--border-soft);
    transition: transform .12s ease, box-shadow .12s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px var(--shadow-2);
}
div[data-testid="stMetricLabel"] { opacity: .65; font-size: .8rem; }
div[data-testid="stMetricValue"] {
    font-size: clamp(0.95rem, 1.6vw, 1.6rem);
    overflow-wrap: anywhere;
    line-height: 1.2;
}

/* ---- buttons ---- */
.stButton > button, .stDownloadButton > button {
    border-radius: 9px;
    font-weight: 500;
    transition: transform .1s ease, box-shadow .15s ease;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    box-shadow: 0 2px 8px rgba(42,120,214,.28);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
}

/* ---- tabs: pill-style, clearer active state ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border-soft);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 500;
    transition: background .12s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--panel); }
.stTabs [aria-selected="true"] {
    background: var(--panel-hover);
    color: var(--acc) !important;
}

/* ---- expanders ---- */
div[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid var(--border-soft) !important;
}
div[data-testid="stExpander"] details summary {
    font-weight: 500;
    border-radius: 12px;
}
div[data-testid="stExpander"] details summary:hover {
    color: var(--acc);
}

/* ---- dataframes / tables ---- */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ---- quote cards in the phrase detail panel ---- */
.quote-card {
    border-left: 3px solid var(--acc);
    background: var(--panel);
    border-radius: 0 10px 10px 0;
    padding: 0.55rem 0.9rem;
    margin: 0.35rem 0;
    transition: box-shadow .12s ease;
}
.quote-card:hover { box-shadow: 0 2px 8px var(--shadow-2); }
.quote-card .quote-product {
    font-weight: 600;
    font-size: 0.82em;
    opacity: 0.75;
    margin-bottom: 0.1rem;
}

/* ---- sidebar pipeline stepper ---- */
.pl-stepper { display: flex; flex-direction: column; gap: 2px; margin: .3rem 0 .1rem; }
.pl-step { display: flex; align-items: center; gap: .55rem; padding: 3px 0; position: relative; }
.pl-step .dot {
    width: 20px; height: 20px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .7rem; font-weight: 700; flex-shrink: 0;
    background: var(--panel); color: color-mix(in srgb, currentColor 55%, transparent);
    border: 1.5px solid var(--border-strong);
}
.pl-step.done .dot { background: var(--acc); border-color: var(--acc); color: #fff; }
.pl-step .label { font-size: .86rem; opacity: .85; }
.pl-step.done .label { opacity: 1; font-weight: 500; }
.pl-step:not(:last-child)::before {
    content: ""; position: absolute; left: 9.5px; top: 24px; width: 1.5px; height: 14px;
    background: var(--border-strong);
}
.pl-step.done:not(:last-child)::before { background: var(--acc); }

/* ---- hero header ---- */
.hero-bar {
    height: 4px; border-radius: 4px; margin-bottom: .9rem;
    background: linear-gradient(90deg, var(--acc), rgba(42,120,214,.25));
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def load_config() -> dict:
    p = ROOT / "config.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def product_dirs() -> list[Path]:
    base = ROOT / "products"
    base.mkdir(exist_ok=True)
    names = {d.name for d in base.iterdir() if d.is_dir()}
    # a fresh (ephemeral) container has an empty local products/ — the
    # line names themselves still have to come from R2 so the sidebar
    # selectbox lists them; the actual files sync down lazily on pick
    names |= set(r2_sync.list_remote_lines(base, ROOT))
    for name in names:
        (base / name).mkdir(exist_ok=True)
    return sorted(base / n for n in names)


def load_mapping(folder: Path) -> dict:
    p = folder / "products.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def save_mapping(folder: Path, mapping: dict) -> None:
    p = folder / "products.yaml"
    p.write_text(
        yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    r2_sync.upload_file(p, ROOT)


def load_overrides(folder: Path) -> dict:
    p = folder / "overrides.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_overrides(folder: Path, ov: dict) -> None:
    ov = {k: v for k, v in ov.items() if v}  # drop empty sections
    p = folder / "overrides.json"
    if ov:
        p.write_text(json.dumps(ov, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        r2_sync.upload_file(p, ROOT)
    elif p.exists():
        p.unlink()
        r2_sync.delete_file(p, ROOT)


def load_gate_labels(folder: Path) -> dict:
    """Мітки ✓/✗ для вето гейта (навчальні дані майбутньої моделі-тріажера).
    Ключ — category|normalize(phrase)|normalize(into), тож мітки переживають
    перезапис review_queue.json наступними прогонами."""
    p = folder / "gate_labels.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_gate_labels(folder: Path, labels: dict) -> None:
    p = folder / "gate_labels.json"
    if labels:
        p.write_text(json.dumps(labels, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        r2_sync.upload_file(p, ROOT)
    elif p.exists():
        p.unlink()
        r2_sync.delete_file(p, ROOT)
    # keep the shared cross-product rule-weight file (products root) current
    # so a label here immediately affects every product's next run
    rebuild_shared_weights(folder.parent)
    r2_sync.upload_file(folder.parent / "gate_rule_weights.json", ROOT)


def gate_label_key(a: dict) -> str:
    return " | ".join((a.get("category", ""),
                       normalize(a.get("phrase", "")),
                       normalize(a.get("into", ""))))


def group_exists(tax: Taxonomy, name: str) -> bool:
    return any(normalize(g.name) == normalize(name) for g in tax.groups.values())


def canonical_exists(tax: Taxonomy, text: str) -> bool:
    return any(normalize(c.text) == normalize(text) for c in tax.canonicals.values())


@st.cache_data(show_spinner=False)
def _phrase_counts(path: str, mtime_ns: int, size: int) -> dict:
    """Cached body of load_phrase_counts. `mtime_ns`/`size` are never read —
    they ARE the cache key, so a rewritten phrases.json (a new pipeline run)
    misses the cache and gets recomputed."""
    counts: dict = {}
    for ph in json.loads(Path(path).read_text(encoding="utf-8")):
        key = (ph["category"], ph["product"], normalize(ph["quote"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_phrase_counts(folder: Path) -> dict:
    """(category, product, normalized quote) -> скільки відгуків так сказали.
    Джерело — phrases.json (сирі витягнуті фрази до злиття).

    The quotes panel calls this on every rerun and the file is ~0.5MB /
    2000+ phrases, each one normalize()d — hence the cache."""
    p = folder / "phrases.json"
    if not p.exists():
        return {}
    s = p.stat()
    return _phrase_counts(str(p), s.st_mtime_ns, s.st_size)


def split_quote_variants(entries: list[str]) -> list[str]:
    """quotes[product] зберігає до 3 сирих варіантів, з'єднаних '; ' —
    розбиваємо назад на окремі цитати без дублікатів."""
    out: list[str] = []
    for e in entries:
        for part in e.split("; "):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def taxonomy_db(folder: Path):
    """DB handle for a folder that has been run at least once, else None
    (avoids creating an empty scoring.db just by browsing the GUI)."""
    if (folder / PRODUCT_DB_NAME).exists():
        return product_db(folder)
    return None


def regenerate_excel(folder: Path) -> Path:
    """--excel-only: scoring.db + overrides.json -> .xlsx"""
    tax = product_db(folder).load_taxonomy()
    apply_overrides(tax, folder / "overrides.json")
    mapping = load_mapping(folder)
    products = {v["name"]: v.get("link", "") for v in mapping.values()}
    out = folder / f"{folder.name} - {date.today().strftime('%d.%m.%Y')}.xlsx"
    out = save_workbook(tax, products, out)
    r2_sync.upload_file(out, ROOT)
    return out


def excel_download(path: Path, key: str) -> None:
    st.download_button(
        f"📥 Завантажити {path.name}", data=path.read_bytes(),
        file_name=path.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key)


def render_phrase_detail(tax: Taxonomy, c, all_products: list[str]) -> None:
    """Одна канонічна фраза: голоси по товарах + всі сирі цитати,
    які було злито в неї (⭐ = збігається з канонічним формулюванням).
    Детальний перегляд по одному товару — на вкладці «Фрази товару»."""
    st.markdown(f"#### 💬 «{html.escape(c.text)}»")
    used_products = [p for p in all_products if c.votes.get(p, 0)]
    st.caption(" · ".join(f"{p} — **{c.votes.get(p, 0)}**" for p in used_products)
               + f" · разом: **{c.total}**")
    for p in used_products:
        for q in split_quote_variants(c.quotes.get(p, [])):
            mark = " ⭐" if normalize(q) == normalize(c.text) else ""
            st.markdown(
                f'<div class="quote-card"><div class="quote-product">'
                f'{html.escape(p)}{mark}</div>{html.escape(q)}</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------- USP board (tab «Дошка»)
# Every board action mutates scoring.db AND records an overrides.json rule
# (pipeline/manual_edits.py) — the fix shows up instantly and is replayed after
# every future pipeline run.
BOARD_UNDO = "board_undo"
BOARD_UNDO_DEPTH = 8


def board_payload(tax: Taxonomy, category: str, min_votes: int = 0,
                  product: str | None = None,
                  inspect: str | None = None) -> dict:
    """Board data for one category: columns = USP groups, cards = rows."""
    products = sorted({p for c in tax.canonicals.values() for p in c.votes})
    groups = []
    # picking a product means "show me this product's board": a row this product
    # never voted for is noise, so the implicit floor is one vote
    floor = max(min_votes, 1) if product else min_votes
    for g in sorted(tax.groups_for(category), key=lambda g: -g.total(tax)):
        all_rows = sorted(g.canonicals(tax), key=lambda c: -c.total)
        rows = []
        for c in all_rows:
            shown = c.votes.get(product, 0) if product else c.total
            if shown < floor:
                continue
            rows.append({"id": c.id, "text": c.text, "total": c.total,
                         "votes": {p: n for p, n in c.votes.items() if n}})
        # a group filtered down to nothing still gets a column — it has to stay
        # a drop target, otherwise the filter hides where you want to drag TO
        groups.append({"id": g.id, "name": g.name,
                       "usage_category": g.usage_category,
                       "total": g.total(tax), "rows": rows,
                       "hidden": len(all_rows) - len(rows)})
    # usage bands exist only on subbucket categories — the component hides the
    # whole chip/menu elsewhere instead of offering a field Excel ignores
    has_bucket = domain_mod.active().has_subbucket(category)
    return {"products": products, "groups": groups, "inspect": inspect or "",
            "product": product or "", "has_bucket": has_bucket,
            "buckets": me.usage_buckets(tax, category) if has_bucket else []}


def board_push_undo(tax: Taxonomy, ov: dict, label: str) -> None:
    """Snapshot taxonomy AND overrides — an edit writes both, so undo has to
    restore both. Compressed and depth-capped: this app has been OOM-killed on
    a 1GB host before."""
    blob = zlib.compress(json.dumps(
        {"tax": me.taxonomy_to_dict(tax), "ov": ov},
        ensure_ascii=False).encode("utf-8"))
    stack = st.session_state.setdefault(BOARD_UNDO, [])
    stack.append((label, blob))
    del stack[:-BOARD_UNDO_DEPTH]


def board_save(folder: Path, db, tax: Taxonomy, ov: dict) -> None:
    reconcile_votes(tax)            # one review = one vote, on every write path
    me.prune_overrides(ov)
    db.save_taxonomy(tax)
    save_overrides(folder, ov)      # uploads overrides.json itself
    r2_sync.upload_file(folder / PRODUCT_DB_NAME, ROOT)


def board_edit(folder: Path, db, label: str, op_fn, auto_merge: bool) -> None:
    """Run one board operation against a freshly loaded taxonomy.

    `op_fn(tax, ov)` must return (note, focus_group_ids, focus_row_ids). The
    result goes to session_state["board_msg"]; the caller reruns."""
    tax = db.load_taxonomy()
    ov = load_overrides(folder)
    board_push_undo(tax, ov, label)
    try:
        note, gids, rids = op_fn(tax, ov)
    except me.EditError as e:
        st.session_state[BOARD_UNDO].pop()
        st.session_state["board_msg"] = {"error": str(e)}
        return
    auto_notes: list[str] = []
    if auto_merge:
        for gid in dict.fromkeys(gids):
            auto_notes += me.auto_merge_group(tax, ov, gid,
                                              focus_ids=list(rids) or None)
    removed = [a for a in prune_empty(tax) if "порожню групу" in a]
    board_save(folder, db, tax, ov)
    st.session_state["board_msg"] = {
        "note": note, "auto": auto_notes, "removed": removed,
        "rows": [r for r in rids if r in tax.canonicals],
        "group": gids[0] if gids else "",
    }


def board_undo(folder: Path, db) -> None:
    stack = st.session_state.get(BOARD_UNDO) or []
    if not stack:
        return
    label, blob = stack.pop()
    state = json.loads(zlib.decompress(blob).decode("utf-8"))
    tax = me.taxonomy_from_dict(state["tax"])
    ov = state["ov"]
    db.save_taxonomy(tax)
    save_overrides(folder, ov)
    r2_sync.upload_file(folder / PRODUCT_DB_NAME, ROOT)
    st.session_state["board_msg"] = {"note": f"↩ Скасовано: {label}"}


def clear_quote_selection(row_id: str) -> None:
    """Checkbox keys are positional (row · product · index), so a `True` left
    over after the picked quotes leave the row would silently select whatever
    slid into their place — every quote action clears the whole selection."""
    prefix = f"bd_qs_{row_id}_"
    for k in [k for k in st.session_state if k.startswith(prefix)]:
        st.session_state.pop(k, None)


def board_quotes_panel(folder: Path, db, cat: str, c) -> None:
    """What a board row is actually made of — the raw review quotes merged into
    it. Same view as the «Фрази товару» tab, but for every product at once."""
    counts = load_phrase_counts(folder)
    # quote_sources pairs every review with the quote it actually said —
    # complete, unlike the ≤3 samples in `quotes`. Rows saved before it
    # existed fall back to the samples and cannot be moved.
    movable: dict[str, list[dict]] = {}
    for item in me.quote_rows(c):
        movable.setdefault(item["product"], []).append(item)
    # one stable checkbox key per card, built BEFORE the widgets: the action
    # bar sits above them and reads the current selection out of
    # session_state, where Streamlit keeps widget values between reruns
    keys = {(p, i): f"bd_qs_{c.id}_{p}_{i}"
            for p, items in movable.items() for i in range(len(items))}
    checked = [(p, i) for (p, i), k in keys.items() if st.session_state.get(k)]
    # the same wording can sit under several products; a move is row-wide, so
    # the selection is a set of quote TEXTS, not of cards
    picked = list(dict.fromkeys(movable[p][i]["quote"] for p, i in checked))

    with st.container(border=True):
        h1, h2 = st.columns([5, 1])
        h1.markdown(f"#### 🔍 «{c.text}»")
        h1.caption(f"{c.total} голос(ів) у цій фразі")
        if h2.button("✕ Закрити", key="bd_ins_close", width="stretch"):
            clear_quote_selection(c.id)
            st.session_state.pop("board_inspect", None)
            st.rerun(scope="fragment")   # nothing written — redraw the board only
        st.caption("Сирі цитати з відгуків, злиті в цю фразу (×N — скільки "
                   "відгуків так сказали, ⭐ — збігається з канонічним "
                   "формулюванням). ↗ — винести цитату в іншу фразу або "
                   "зробити з неї окремий рядок; голоси її відгуків підуть "
                   "разом з нею. ☐ зліва — виділити кілька і зробити те саме "
                   "гуртом (або прибрати їх з аналізу зовсім).")
        if keys:
            a1, a2, a3, a4, a5 = st.columns([1, 1, 1.7, 1.7, 1.6])
            if a1.button("☑ Усі", key=f"bd_qall_{c.id}", width="stretch",
                         disabled=len(checked) == len(keys)):
                for k in keys.values():
                    st.session_state[k] = True
                st.rerun(scope="fragment")   # selection only, nothing written
            if a2.button("☐ Зняти", key=f"bd_qnone_{c.id}", width="stretch",
                         disabled=not checked):
                clear_quote_selection(c.id)
                st.rerun(scope="fragment")   # selection only, nothing written
            n = len(picked)
            if a3.button(f"↗ Перенести ({n})", key=f"bd_qmv_{c.id}",
                         width="stretch", disabled=not n,
                         help="В іншу фразу, окремими рядками або одним "
                              "новим рядком"):
                st.session_state["board_quote_move"] = {"row": c.id,
                                                        "quotes": picked}
                st.rerun()
            if a4.button(f"🚫 Без USP ({n})", key=f"bd_qpark_{c.id}",
                         width="stretch", disabled=not n,
                         help=f"Кожна цитата — окремим рядком у групі "
                              f"«{me.NO_USP_GROUP}» цієї категорії"):
                board_edit(
                    folder, db, f"винесення цитат без USP ({n})",
                    lambda tax, ov: (
                        me.move_quotes(tax, ov, c.id, picked,
                                       group_name=me.NO_USP_GROUP,
                                       each_own_row=True),
                        [], []),
                    False)   # never auto-merge: rows just split off would be
                             # folded straight back by the gate that grouped them
                clear_quote_selection(c.id)
                st.rerun()
            if a5.button(f"🗑 Прибрати ({n})", key=f"bd_qdrop_{c.id}",
                         width="stretch", disabled=not n,
                         help="Прибрати цитати з аналізу разом з їхніми "
                              "голосами"):
                st.session_state["board_quote_drop"] = {"row": c.id,
                                                        "quotes": picked}
                st.rerun()
        for p, n in sorted(c.votes.items(), key=lambda kv: -kv[1]):
            if not n:
                continue
            st.markdown(f"**{p}** — {n} голос(ів)")
            if movable.get(p):
                for i, item in enumerate(movable[p]):
                    mark = (" ⭐" if normalize(item["quote"]) == normalize(c.text)
                            else "")
                    # the ☐ column was 0.5 wide — enough for the glyph, not
                    # enough to hit comfortably next to a wall of quote text
                    q0, q1, q2 = st.columns([0.8, 8.0, 1.2])
                    q0.checkbox("Обрати цитату", key=keys[(p, i)],
                                label_visibility="collapsed")
                    q1.markdown(
                        f'<div class="quote-card"><div class="quote-product">'
                        f'×{item["votes"]}{mark}</div>'
                        f'{html.escape(item["quote"])}</div>',
                        unsafe_allow_html=True)
                    if q2.button("↗", key=f"bd_mq_{c.id}_{p}_{i}",
                                 help="Перенести цю цитату в іншу фразу "
                                      "або в окремий рядок"):
                        st.session_state["board_quote_move"] = {
                            "row": c.id, "quotes": [item["quote"]]}
                        st.rerun()
                continue
            variants = split_quote_variants(c.quotes.get(p, []))
            if not variants:
                st.caption("Сирі цитати для цього товару не збереглися.")
                continue
            covered = 0
            for v in variants:
                k = counts.get((cat, p, normalize(v)), 0)
                covered += k or 1
                mark = " ⭐" if normalize(v) == normalize(c.text) else ""
                st.markdown(
                    f'<div class="quote-card"><div class="quote-product">'
                    f'×{k or "?"}{mark}</div>{html.escape(v)}</div>',
                    unsafe_allow_html=True)
            if covered < n:
                st.caption(f"⚠️ Показані цитати покривають {covered} з {n} "
                           f"голосів — решта варіантів не збереглася "
                           f"(зберігається до 3 на злиття).")


def quote_preview(quotes: list[str], limit: int = 5) -> None:
    for q in quotes[:limit]:
        st.markdown(f'<div class="quote-card">{html.escape(q)}</div>',
                    unsafe_allow_html=True)
    if len(quotes) > limit:
        st.caption(f"…і ще {len(quotes) - limit} цитат(и).")


@st.dialog("↗ Перенести цитати")
def board_move_quotes_dialog(folder: Path, db, spec: dict):
    """One quote (the ↗ on a card) or a whole selection — the same dialog; the
    only difference is that a batch can also go each-into-its-own-row."""
    tax = db.load_taxonomy()
    src = tax.canonicals.get(spec["row"])
    quotes = [q for q in spec.get("quotes", []) if q]
    if src is None or not quotes:
        st.warning("Рядка вже немає — оновіть сторінку.")
        if st.button("Закрити", key="bd_mq_close"):
            st.session_state.pop("board_quote_move", None)
            st.rerun()
        return
    cat = me.category_of(tax, src)
    n = len(quotes)
    quote_preview(quotes)
    st.caption(("Зараз ця цитата рахується" if n == 1
                else f"Зараз ці {n} цитат(и) рахуються")
               + f" в рядку «{src.text}». Голоси відгуків, які сказали саме "
                 + ("її" if n == 1 else "їх") + ", підуть за "
                 + ("нею." if n == 1 else "ними."))
    SAME = "🔗 В іншу фразу"
    OWN = "🆕 Залишити окремим рядком" if n == 1 else "🆕 Окремими рядками"
    JOINT = "🧩 Усі в один новий рядок"
    PARK = f"🚫 Без USP (група «{me.NO_USP_GROUP}»)"
    modes = [SAME, OWN] + ([JOINT] if n > 1 else []) + [PARK]
    mode = st.radio("Куди перенести?", modes, key=f"bd_mq_mode_{n > 1}")
    target_id = new_text = group_id = None
    group_name, each_own = "", False
    if mode == SAME:
        others = sorted((c for c in tax.canonicals.values()
                         if c.id != src.id and me.category_of(tax, c) == cat),
                        key=lambda c: -c.total)
        if not others:
            st.warning("У цій категорії немає інших фраз.")
        else:
            def _label(cid: str) -> str:
                c = tax.canonicals[cid]
                g = tax.groups.get(c.group_id)
                return f"{g.name if g else '?'} · {c.text} (Σ{c.total})"
            target_id = st.selectbox("Фраза, у яку перенести",
                                     [c.id for c in others],
                                     format_func=_label, key="bd_mq_to")
    elif mode == PARK:
        each_own, group_name = True, me.NO_USP_GROUP
        st.caption(f"Кожна цитата стане окремим рядком у групі "
                   f"«{me.NO_USP_GROUP}» цієї категорії (створимо її, якщо "
                   "ще немає). В Excel ця група пишеться як звичайна USP — "
                   "просто ці голоси більше не додаються до реальних USP.")
    else:
        # one quote always gets an editable wording; a batch keeps each quote
        # verbatim unless the user asked for one joint row
        if mode == JOINT or n == 1:
            new_text = st.text_input("Формулювання нового рядка",
                                     value=quotes[0][:120], key="bd_mq_text")
        else:
            each_own = True
            st.caption("Кожна цитата стане окремим рядком зі своїм "
                       "формулюванням.")
        groups = sorted(tax.groups_for(cat), key=lambda g: -g.total(tax))
        ids = [g.id for g in groups]
        cur = ids.index(src.group_id) if src.group_id in ids else 0
        group_id = st.selectbox(
            "USP для нових рядків", ids, index=cur,
            format_func=lambda i: tax.groups[i].name, key="bd_mq_grp")
    c1, c2 = st.columns(2)
    ok = c1.button("↗ Перенести", type="primary", width="stretch",
                   key="bd_mq_ok", disabled=mode == SAME and not target_id)
    if ok:
        label = (f"перенесення цитати «{quotes[0][:40]}»" if n == 1
                 else f"перенесення цитат ({n})")
        board_edit(
            folder, db, label,
            lambda tax, ov: (
                me.move_quotes(tax, ov, spec["row"], quotes,
                               target_row_id=target_id, new_text=new_text,
                               group_id=group_id, group_name=group_name,
                               each_own_row=each_own),
                [], []),
            False)   # never auto-merge here: a row just split off would be
                     # folded straight back by the gate that grouped it
        clear_quote_selection(spec["row"])
        st.session_state.pop("board_quote_move", None)
        st.rerun()
    if c2.button("Скасувати", width="stretch", key="bd_mq_cancel"):
        st.session_state.pop("board_quote_move", None)
        st.rerun()


@st.dialog("🗑 Прибрати цитати")
def board_drop_quotes_dialog(folder: Path, db, spec: dict):
    tax = db.load_taxonomy()
    src = tax.canonicals.get(spec["row"])
    quotes = [q for q in spec.get("quotes", []) if q]
    if src is None or not quotes:
        st.warning("Рядка вже немає — оновіть сторінку.")
        if st.button("Закрити", key="bd_dq_close"):
            st.session_state.pop("board_quote_drop", None)
            st.rerun()
        return
    n = len(quotes)
    quote_preview(quotes)
    st.warning(f"Ці цитати ({n}) буде прибрано з аналізу: голоси відгуків, "
               f"які сказали саме їх, зникнуть з рядка «{src.text}» і нікуди "
               "не перейдуть. Відгук, чиї ІНШІ цитати лишаються в рядку, свій "
               "голос зберігає. Правило запишеться в overrides.json, тож ці "
               "цитати будуть прибрані й на наступних прогонах. Скасувати — "
               "кнопкою ↩ над дошкою.")
    c1, c2 = st.columns(2)
    if c1.button("🗑 Прибрати", type="primary", width="stretch",
                 key="bd_dq_ok"):
        board_edit(
            folder, db, f"прибирання цитат ({n})",
            lambda tax, ov: (
                me.drop_quotes(tax, ov, spec["row"], quotes), [], []),
            False)
        clear_quote_selection(spec["row"])
        st.session_state.pop("board_quote_drop", None)
        st.rerun()
    if c2.button("Скасувати", width="stretch", key="bd_dq_cancel"):
        st.session_state.pop("board_quote_drop", None)
        st.rerun()


def board_close_dialog() -> None:
    st.session_state.pop("board_pending", None)


@st.dialog("🔀 Злити USP")
def board_merge_groups_dialog(folder: Path, db, op: dict, auto_merge: bool):
    tax = db.load_taxonomy()
    src, tgt = tax.groups.get(op["source"]), tax.groups.get(op["target"])
    if src is None or tgt is None:
        st.warning("Однієї з груп уже немає — оновіть сторінку.")
        if st.button("Закрити", key="bd_mg_close"):
            board_close_dialog()
            st.rerun()
        return
    st.markdown(f"**«{src.name}»** ({src.total(tax)} гол.) вливається в "
                f"**«{tgt.name}»** ({tgt.total(tax)} гол.)")
    st.caption(f"Разом стане {len(src.canonicals(tax)) + len(tgt.canonicals(tax))} "
               f"фраз · {src.total(tax) + tgt.total(tax)} голосів.")
    options = [tgt.name] + ([src.name] if src.name != tgt.name else [])
    CUSTOM = "✏️ Інша назва…"
    pick = st.radio("Яку назву залишити?", options + [CUSTOM], key="bd_mg_pick")
    name = pick
    if pick == CUSTOM:
        name = st.text_input("Нова назва USP", value=tgt.name, key="bd_mg_new")
    c1, c2 = st.columns(2)
    if c1.button("🔀 Злити", type="primary", width="stretch", key="bd_mg_ok"):
        board_edit(
            folder, db, f"злиття USP «{src.name}»+«{tgt.name}»",
            lambda tax, ov: (
                me.merge_groups(tax, ov, [op["source"]], op["target"],
                                name=name),
                [op["target"]], []),
            auto_merge)
        board_close_dialog()
        st.rerun()
    if c2.button("Скасувати", width="stretch", key="bd_mg_cancel"):
        board_close_dialog()
        st.rerun()


@st.dialog("🔗 Злити фрази")
def board_merge_rows_dialog(folder: Path, db, op: dict, auto_merge: bool):
    tax = db.load_taxonomy()
    keep = tax.canonicals.get(op["into"])
    others = me.rows_by_id(tax, [r for r in op["rows"] if r != op["into"]])
    if keep is None or not others:
        st.warning("Рядків уже немає — оновіть сторінку.")
        if st.button("Закрити", key="bd_mr_close"):
            board_close_dialog()
            st.rerun()
        return
    st.markdown("Ці формулювання стануть **одним рядком**; голоси "
                "підсумуються (один відгук = один голос).")
    for c in [keep] + others:
        st.markdown(f'<div class="quote-card"><div class="quote-product">'
                    f'{c.total} гол.</div>{html.escape(c.text)}</div>',
                    unsafe_allow_html=True)
    texts = list(dict.fromkeys([keep.text] + [c.text for c in others]))
    final = st.radio("Яке формулювання залишити?", texts, key="bd_mr_pick")
    c1, c2 = st.columns(2)
    if c1.button("🔗 Злити", type="primary", width="stretch", key="bd_mr_ok"):
        board_edit(
            folder, db, f"злиття рядків у «{final}»",
            lambda tax, ov: (
                me.merge_rows(tax, ov, op["rows"], op["into"], keep_text=final),
                [keep.group_id], [op["into"]]),
            auto_merge)
        board_close_dialog()
        st.rerun()
    if c2.button("Скасувати", width="stretch", key="bd_mr_cancel"):
        board_close_dialog()
        st.rerun()


@st.dialog("➕ Нова USP")
def board_new_group_dialog(folder: Path, db, op: dict, auto_merge: bool):
    tax = db.load_taxonomy()
    rows = me.rows_by_id(tax, op["rows"])
    if not rows:
        st.warning("Фраз уже немає — оновіть сторінку.")
        if st.button("Закрити", key="bd_ng_close"):
            board_close_dialog()
            st.rerun()
        return
    st.caption("Фрази, що переїдуть у нову USP:")
    for c in rows:
        st.markdown(f"- «{c.text}» — {c.total} гол.")
    name = st.text_input("Назва USP", value=rows[0].text[:60], key="bd_ng_name")
    c1, c2 = st.columns(2)
    if c1.button("➕ Створити", type="primary", width="stretch", key="bd_ng_ok"):
        holder: dict = {}

        def _op(tax, ov):
            note = me.new_group_from_rows(tax, ov, op["rows"], name)
            row = tax.canonicals.get(op["rows"][0])
            holder["gid"] = row.group_id if row else ""
            return note, [holder["gid"]], list(op["rows"])

        board_edit(folder, db, f"нова USP «{name}»", _op, auto_merge)
        board_close_dialog()
        st.rerun()
    if c2.button("Скасувати", width="stretch", key="bd_ng_cancel"):
        board_close_dialog()
        st.rerun()


def load_usage_history() -> pd.DataFrame:
    """All logged pipeline runs across every product line, oldest first."""
    rows = root_db(ROOT).load_usage()
    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "date", "product_line", "provider",
            "extract_model", "group_model", "reviews", "phrases", "groups",
            "calls", "cache_hits", "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_write_tokens", "cost_usd", "cost_known"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def append_usage_history(record: dict) -> None:
    root_db(ROOT).append_usage(record)


def folder_stats(folder: Path) -> dict:
    """Quick at-a-glance counters for the header status row."""
    pdfs = list(folder.glob("*.pdf"))
    reviews_n = groups_n = phrases_n = 0
    if (folder / "reviews.json").exists():
        reviews_n = len(json.loads((folder / "reviews.json").read_text(encoding="utf-8")))
    db = taxonomy_db(folder)
    if db is not None:
        groups_n, phrases_n = db.taxonomy_counts()
    xlsx = sorted(folder.glob("*.xlsx"), reverse=True)
    return {
        "pdfs": len(pdfs), "reviews": reviews_n,
        "groups": groups_n, "phrases": phrases_n,
        "last_xlsx": xlsx[0].name if xlsx else None,
    }


# ---------------------------------------------------------------- sidebar
r2_sync.sync_file_down(ROOT / ROOT_DB_NAME, ROOT)  # before anything can open/create it locally
cfg = load_config()
# cap on concurrent LLM calls (asyncio semaphore shared by all passes)
set_max_concurrency(cfg.get("max_concurrent_requests", 10))
# optional deterministic synonym families from config.yaml — must be
# installed before any grouping/row-merge pass
from pipeline.similarity import set_synonym_families, set_attribute_families
set_synonym_families(cfg.get("synonym_families"))
set_attribute_families(cfg.get("attribute_families"))

with st.sidebar:
    st.title("📊 Review Scoring")
    st.caption("PDF з відгуками → Excel за SOP")
    st.divider()

    dirs = product_dirs()
    names = [d.name for d in dirs]
    picked = st.selectbox("Продуктова лінійка", names or ["(немає)"],
                          disabled=not names)
    folder = ROOT / "products" / picked if names else None
    if folder is not None:
        r2_sync.sync_folder_down(folder, ROOT)

    # per-product domain profile (categories + Excel layout); absent =>
    # built-in default. Installed before any extraction/grouping/excel pass.
    _domain_err = None
    try:
        domain_mod.set_active_domain(domain_mod.load_domain(folder))
    except (ValueError, KeyError) as e:
        domain_mod.set_active_domain(None)
        _domain_err = str(e)
    if _domain_err:
        st.error(f"Помилка в domain.yaml: {_domain_err}\n\nВикористано "
                 "профіль за замовчуванням.")

    if folder is not None:
        # GUI create/edit for domain.yaml — Streamlit Cloud has no shell and
        # products/ lives only in R2 (gitignored locally too), so
        # `python run.py ... --init-domain` is unreachable there. This is
        # the same write_default_yaml() the CLI flag calls, just triggered
        # from a button and pushed to R2 instead of touched by hand on disk.
        _domain_path = folder / "domain.yaml"
        _has_domain = _domain_path.exists()
        _label = ("⚙️ Профіль домену (domain.yaml)" if _has_domain and not _domain_err
                  else "⚠️ Профіль домену — не налаштовано")
        with st.expander(_label):
            if not _has_domain:
                st.info("Немає власного профілю категорій — пайплайн "
                        "працює на дефолтному (косметика першої допомоги): "
                        "чужі категорії, заголовки й приклади для судді "
                        "групування. Створіть свій — без доступу до "
                        "файлової системи чи git, прямо тут.")
                if st.button("Створити профіль за замовчуванням",
                             key="create_domain", width="stretch"):
                    domain_mod.write_default_yaml(_domain_path)
                    r2_sync.upload_file(_domain_path, ROOT)
                    st.rerun()
            else:
                if _domain_err:
                    st.error(f"Некоректний YAML: {_domain_err}")
                st.caption(str(_domain_path))
                current_text = _domain_path.read_text(encoding="utf-8")
                edited = st.text_area(
                    "Категорії, аркуші, приклади для судді (judge_examples) "
                    "— редагуйте прямо тут",
                    value=current_text, height=320, key="domain_yaml_editor")
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    if st.button("💾 Зберегти", key="save_domain",
                                 width="stretch"):
                        try:
                            data = yaml.safe_load(edited) or {}
                            domain_mod.domain_from_dict(data)  # validate
                        except Exception as e:
                            st.error(f"Не збережено — некоректний "
                                     f"domain.yaml: {e}")
                        else:
                            _domain_path.write_text(edited, encoding="utf-8")
                            r2_sync.upload_file(_domain_path, ROOT)
                            st.success("Збережено.")
                            st.rerun()
                with dcol2:
                    if st.button("↺ Скинути на дефолт", key="reset_domain",
                                 width="stretch"):
                        domain_mod.write_default_yaml(_domain_path)
                        r2_sync.upload_file(_domain_path, ROOT)
                        st.rerun()

    with st.expander("➕ Нова лінійка"):
        new_name = st.text_input("Назва папки", placeholder="styptic")
        if st.button("Створити", width="stretch") and new_name.strip():
            (ROOT / "products" / new_name.strip()).mkdir(parents=True, exist_ok=True)
            st.rerun()

    if folder is not None:
        with st.expander("🗑 Видалити лінійку"):
            st.warning(f"Незворотно видалить **{folder.name}** — усі PDF, "
                       "таксономію, Excel і кеш локально та в хмарі (R2).")
            confirm = st.text_input(
                "Введіть назву папки для підтвердження",
                key=f"del_confirm_{folder.name}", placeholder=folder.name)
            if st.button("Видалити назавжди", type="primary",
                         width="stretch", disabled=confirm != folder.name):
                close_product_db(folder)
                r2_sync.delete_folder(folder, ROOT)
                try:
                    shutil.rmtree(folder)
                except OSError as e:
                    st.error(
                        f"Видалено в хмарі (R2), але локальні файли "
                        f"лишились заблокованими: {e}. Закрийте інші "
                        "вкладки/сесії цієї лінійки і спробуйте ще раз.")
                    st.stop()
                st.rerun()

    if folder is not None:
        st.divider()
        st.caption("**Стан пайплайна**")
        stats = folder_stats(folder)
        steps = [
            ("PDF завантажено", stats["pdfs"] > 0),
            ("Відгуки розпарсено", stats["reviews"] > 0),
            ("Таксономія побудована", stats["groups"] > 0),
            ("Excel готовий", stats["last_xlsx"] is not None),
        ]
        rows = "".join(
            f'<div class="pl-step{" done" if done else ""}">'
            f'<div class="dot">{"✓" if done else i + 1}</div>'
            f'<div class="label">{label}</div></div>'
            for i, (label, done) in enumerate(steps))
        st.markdown(f'<div class="pl-stepper">{rows}</div>', unsafe_allow_html=True)

    st.divider()
    st.caption(f"Провайдер: **{cfg.get('provider', 'anthropic')}**")
    st.caption(f"Модель (групування): **{cfg.get('model', '—')}**")
    st.caption(f"Модель (екстракція): **{cfg.get('extract_model') or cfg.get('model', '—')}**")

if folder is None:
    st.info("Створіть продуктову лінійку в бічній панелі, щоб почати.")
    st.stop()

# ---------------------------------------------------------------- header
st.markdown('<div class="hero-bar"></div>', unsafe_allow_html=True)
st.title(f"📊 {folder.name}")
stats = folder_stats(folder)
h1, h2, h3, h4 = st.columns(4)
h1.metric("PDF", stats["pdfs"])
h2.metric("Відгуків", stats["reviews"] or "—")
h3.metric("USP-груп", stats["groups"] or "—")
h4.metric("Останній Excel", stats["last_xlsx"] or "—")
st.divider()

(tab_prod, tab_run, tab_res, tab_board, tab_by_prod, tab_review, tab_fix,
 tab_cost) = st.tabs(
    ["📦 Продукти й PDF", "▶️ Запуск", "🗂 Результати", "🧩 Дошка USP",
     "🔎 Фрази товару", "🕵 Перевірити", "✏️ Корекції", "💰 Витрати"])


# ---------------------------------------------------------------- products
with tab_prod:
    with st.container(border=True):
        st.markdown("**📤 Завантажити PDF**")
        up = st.file_uploader("Перетягніть PDF з відгуками сюди", type="pdf",
                              accept_multiple_files=True, label_visibility="collapsed")
        if up and st.button("Зберегти PDF у папку", type="primary"):
            for f in up:
                dest = folder / f.name
                dest.write_bytes(f.getbuffer())
                r2_sync.upload_file(dest, ROOT)
            st.toast(f"Збережено {len(up)} PDF", icon="✅")
            st.rerun()

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        st.info("У папці ще немає PDF — завантажте хоча б один вище.")
    else:
        with st.container(border=True):
            mapping = load_mapping(folder)
            for p in pdfs:  # skeleton for new files
                mapping.setdefault(p.stem, {"name": product_key(p.stem), "link": ""})

            st.markdown(f"**✏️ PDF у папці ({len(pdfs)}): короткі назви та посилання** "
                        "<span style='color:gray;font-weight:normal'>"
                        "(назва — підпис колонки продукту в Excel)</span>",
                        unsafe_allow_html=True)
            rows = [{"PDF": stem, "Назва": v.get("name", ""), "Посилання": v.get("link", "")}
                    for stem, v in mapping.items()]
            edited = st.data_editor(
                pd.DataFrame(rows), hide_index=True, width="stretch",
                disabled=["PDF"], key="prod_editor")
            if st.button("💾 Зберегти products.yaml", type="primary"):
                new_map = {r["PDF"]: {"name": str(r["Назва"]).strip(),
                                      "link": str(r["Посилання"] or "").strip()}
                           for r in edited.to_dict("records")}

                # detect product-name renames (old "Назва" -> new "Назва" for
                # the same PDF stem) so a taxonomy already built under the
                # old name follows the rename instead of silently forking
                # into a second column on the next run.
                rename_pairs: dict[str, str] = {}
                conflicts: set[str] = set()
                for stem, new_v in new_map.items():
                    old_v = mapping.get(stem)
                    if not old_v:
                        continue
                    old_name, new_name = old_v.get("name", ""), new_v["name"]
                    if old_name and new_name and old_name != new_name:
                        prev = rename_pairs.get(old_name)
                        if prev is not None and prev != new_name:
                            conflicts.add(old_name)
                        else:
                            rename_pairs[old_name] = new_name
                for c in conflicts:
                    rename_pairs.pop(c, None)

                save_mapping(folder, new_map)

                if rename_pairs:
                    rn_db = taxonomy_db(folder)
                    if rn_db is not None:
                        rn_tax = rn_db.load_taxonomy()
                        n = rn_tax.remap_products(
                            key_fn=lambda p: rename_pairs.get(p, p))
                        if n:
                            reconcile_votes(rn_tax)
                            rn_db.save_taxonomy(rn_tax)
                            pairs_txt = ", ".join(
                                f"«{o}»→«{n_}»" for o, n_ in rename_pairs.items())
                            st.toast(f"Перейменовано {n} стовпців товару в "
                                     f"таксономії ({pairs_txt})", icon="🔀")
                if conflicts:
                    st.warning(
                        "Неоднозначне перейменування (одна стара назва веде "
                        "до кількох нових) пропущено для: " +
                        ", ".join(f"«{c}»" for c in conflicts) +
                        " — перейменуйте по одному товару за раз.")

                st.toast("products.yaml збережено", icon="💾")


# ---------------------------------------------------------------- run
with tab_run:
    pdfs = sorted(folder.glob("*.pdf"))
    mapping = load_mapping(folder)
    missing = [p.stem for p in pdfs if p.stem not in mapping]

    if not pdfs:
        st.warning("Спершу додайте PDF на вкладці «Продукти й PDF».")
    elif missing:
        st.warning("У products.yaml немає записів для: " + ", ".join(missing) +
                   ". Збережіть таблицю на вкладці «Продукти й PDF».")
    else:
        with st.container(border=True):
            st.markdown("**⚙️ Налаштування прогону**")
            c1, c2, c3 = st.columns(3)
            with c1:
                limit = st.number_input("Ліміт відгуків на PDF (0 = всі)",
                                        min_value=0, value=0, step=5,
                                        help="Швидкий тест — напр. 15")
            with c2:
                st.markdown("<div style='height:1.9em'></div>",
                           unsafe_allow_html=True)
                use_cutoff = st.checkbox("Лише відгуки після дати")
                cutoff = st.date_input("Дата відсічення",
                                       value=date(2025, 1, 1),
                                       disabled=not use_cutoff)
            with c3:
                st.markdown("<div style='height:1.9em'></div>",
                           unsafe_allow_html=True)
                fresh = st.checkbox("🔄 Таксономія з нуля (--fresh)",
                                    help="Інакше нові фрази наповнюють існуючі групи")

            st.markdown("**🤖 Моделі за кроками** (порожньо = з config.yaml)")
            provider_ui = cfg.get("provider", "anthropic")
            choices = MODEL_CHOICES.get(provider_ui, MODEL_CHOICES["anthropic"])

            def model_select(label: str, cfg_key: str, ui_key: str,
                             cfg_default: str = "model") -> str | None:
                """Selectbox for one pipeline role's model override. Returns
                None (=> caller falls back to config.yaml) unless the user
                picked a specific model or typed a custom slug."""
                current = cfg.get(cfg_key) or (cfg.get(cfg_default) if cfg_default else None)
                options = [MODEL_FROM_CONFIG] + choices + [MODEL_CUSTOM]
                sel = st.selectbox(f"{label} (нині: {current or '—'})",
                                   options, key=f"model_sel_{ui_key}")
                if sel == MODEL_CUSTOM:
                    custom = st.text_input("Слаг моделі", key=f"model_custom_{ui_key}")
                    return custom.strip() or None
                if sel == MODEL_FROM_CONFIG:
                    return None
                return sel

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                model_group = model_select("Групування", "model", "group")
            with mc2:
                model_extract = model_select("Екстракція", "extract_model", "extract")
            with mc3:
                model_consolidate = model_select("Консолідація", "consolidate_model", "consolidate")
            with mc4:
                model_verify = model_select("Перевірка нових груп", "model", "verify")
            with mc5:
                model_reassign = model_select("Перепризначення", "reassign_model", "reassign")

            run_clicked = st.button("▶️ Запустити", type="primary", width="stretch")

        if run_clicked:
            try:
                with st.status("Виконується…", expanded=True) as status:
                    products = {v["name"]: v.get("link", "")
                                for v in mapping.values()}

                    # 1. parse
                    st.write("**1/4 Парсинг PDF**")
                    all_reviews = []
                    parse_warnings = []
                    for pdf in pdfs:
                        product = mapping[pdf.stem]["name"]
                        pstats = {}
                        rs = parse_pdf(pdf, product, stats=pstats)
                        if pstats.get("expected", 0) > pstats.get("parsed", 0):
                            st.warning(
                                f"{pdf.name}: розпізнано {pstats['parsed']} з "
                                f"{pstats['expected']} відгуків — втрачені "
                                f"фрагменти у parse_warnings.json")
                            parse_warnings.append({"product": product,
                                                   **pstats})
                        rs = filter_reviews(
                            rs, max_reviews=None,
                            cutoff=cutoff if use_cutoff else None)
                        if limit:
                            rs = rs[:limit]
                        st.write(f"　{product}: {len(rs)} відгуків")
                        all_reviews.extend(rs)
                    pw_path = folder / "parse_warnings.json"
                    if parse_warnings:
                        pw_path.write_text(
                            json.dumps(parse_warnings, ensure_ascii=False,
                                       indent=1), encoding="utf-8")
                    elif pw_path.exists():
                        pw_path.unlink()   # stale warnings from an older run
                    (folder / "reviews.json").write_text(
                        json.dumps([dataclasses.asdict(r) for r in all_reviews],
                                   ensure_ascii=False, indent=1),
                        encoding="utf-8")

                    provider = cfg.get("provider", "anthropic")
                    db = product_db(folder)
                    llm_group = LLM(model=model_group or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("effort", "medium"),
                              provider=provider)
                    llm_extract = LLM(
                              model=model_extract or cfg.get("extract_model") or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("extract_effort") or cfg.get("effort", "medium"),
                              provider=provider)
                    llm_consolidate = LLM(
                              model=model_consolidate or cfg.get("consolidate_model") or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("consolidate_effort") or cfg.get("effort", "medium"),
                              provider=provider)
                    # narrow lexical dedup of new groups — its own (lower) effort
                    llm_verify = LLM(
                              model=model_verify or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("verify_effort") or cfg.get("effort", "medium"),
                              provider=provider)
                    # replay against a FINAL taxonomy (matching, not building) —
                    # its own (lower) effort; reasoning output is the main cost
                    llm_reassign = LLM(
                              model=model_reassign or cfg.get("reassign_model") or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("reassign_effort") or cfg.get("effort", "medium"),
                              provider=provider)

                    # 2. extract
                    st.write(f"**2/5 Екстракція фраз** ({len(all_reviews)} відгуків)")
                    ebar = st.progress(0.0)
                    phrases = extract_phrases(
                        all_reviews, llm_extract,
                        batch_size=cfg.get("extract_batch", 8),
                        progress=lambda d, t: ebar.progress(min(d / max(t, 1), 1.0)))
                    st.write(f"　витягнуто фраз: {len(phrases)}")
                    vv = validate_verbatim(phrases, all_reviews,
                                           llm=llm_extract)
                    if vv["repaired"]:
                        st.write(f"　відновлено дослівність: "
                                 f"{len(vv['repaired'])} цитат")
                    if vv["unverified"]:
                        (folder / "verbatim_warnings.json").write_text(
                            json.dumps(vv["unverified"], ensure_ascii=False,
                                       indent=1), encoding="utf-8")
                        st.warning(f"Вилучено вигадані цитати: "
                                   f"{len(vv['unverified'])} "
                                   f"(verbatim_warnings.json)")
                    phrases, dd = dedupe_overlapping(phrases)
                    if dd["dropped"] or dd["split"]:
                        st.write(f"　усунено перекриття фраз: "
                                 f"{len(dd['dropped'])} прибрано, "
                                 f"{len(dd['split'])} розділено")
                    (folder / "phrases.json").write_text(
                        json.dumps([dataclasses.asdict(p) for p in phrases],
                                   ensure_ascii=False, indent=1),
                        encoding="utf-8")

                    # 3. group
                    st.write("**3/5 Групування**")
                    # ваші ✓/✗ вердикти з вкладки «Перевірити» стають
                    # прецедентами й вагами для вето гейта на цьому прогоні
                    load_gate_precedents(folder, cfg.get("gate_feedback"))
                    audit: list[dict] = []
                    gbar = st.progress(0.0, text="…")
                    tax = Taxonomy() if fresh else db.load_taxonomy()
                    tax = group_phrases(
                        phrases, tax, llm_group,
                        batch_size=cfg.get("group_batch", 25),
                        progress=lambda cat, d, t: gbar.progress(
                            min(d / max(t, 1), 1.0), text=f"{cat}: {d}/{t}"),
                        double_check=cfg.get("double_check", True),
                        audit=audit,
                        verify_llm=llm_verify,
                        checkpoint=lambda: db.save_taxonomy(tax))

                    if cfg.get("consolidate", True):
                        st.write("　консолідація таксономії…")
                        actions = consolidate_taxonomy(tax, llm_consolidate)
                        for a in actions:
                            st.write(f"　· {a}")
                        audit.extend({"type": "consolidate", "category": "",
                                      "action": a} for a in actions)

                    if cfg.get("split", True):
                        st.write("　розбиття мегагруп…")
                        s_actions = split_groups(
                            tax, llm_consolidate,
                            min_share=cfg.get("split_min_share", 0.12),
                            min_rows=cfg.get("split_min_rows", 25),
                            audit=audit)
                        for a in s_actions:
                            st.write(f"　· {a}")

                    # 4. reassignment against the final taxonomy
                    if cfg.get("reassign", True):
                        st.write("**4/5 Перепризначення** (фінальна таксономія)")
                        rbar = st.progress(0.0, text="…")
                        r_actions = reassign_phrases(
                            phrases, tax, llm_reassign,
                            batch_size=cfg.get("group_batch", 25),
                            progress=lambda cat, d, t: rbar.progress(
                                min(d / max(t, 1), 1.0), text=f"{cat}: {d}/{t}"),
                            audit=audit,
                            checkpoint=lambda: db.save_taxonomy(tax))
                        for a in r_actions:
                            st.write(f"　· {a}")

                    if cfg.get("row_merge", True):
                        st.write("　злиття рядків-синонімів…")
                        m_actions = merge_sibling_rows(tax, llm_consolidate,
                                                       audit)
                        for a in m_actions:
                            st.write(f"　· {a}")

                    for g in tax.groups.values():
                        cans = g.canonicals(tax)
                        if len(cans) == 1 and cans[0].total <= 2:
                            audit.append({"type": "singleton_group",
                                          "category": g.category,
                                          "group": g.name, "row": cans[0].text,
                                          "votes": cans[0].total})

                    reconcile_votes(tax)   # persist the one-review-one-vote invariant
                    db.save_taxonomy(tax)
                    apply_overrides(tax, folder / "overrides.json")

                    seen_a: set = set()
                    audit = [a for a in audit
                             if (k := json.dumps(a, ensure_ascii=False,
                                                 sort_keys=True))
                             not in seen_a and not seen_a.add(k)]
                    (folder / "review_queue.json").write_text(
                        json.dumps(audit, ensure_ascii=False, indent=1),
                        encoding="utf-8")
                    if audit:
                        st.info(f"🕵 Місць для перевірки: {len(audit)} — "
                                f"вкладка «Перевірити»")

                    qwarnings = workbook_warnings(tax)
                    if qwarnings:
                        (folder / "quality_warnings.json").write_text(
                            json.dumps(qwarnings, ensure_ascii=False,
                                      indent=1), encoding="utf-8")
                        st.warning("Перевірте вручну:\n" +
                                  "\n".join(f"- {w}" for w in qwarnings))

                    # 5. excel
                    st.write("**5/5 Excel**")
                    out = folder / (f"{folder.name} - "
                                    f"{date.today().strftime('%d.%m.%Y')}.xlsx")
                    out = save_workbook(tax, products, out)

                    status.update(label="Готово ✅", state="complete")

                # one bulk push instead of tracking every intermediate
                # write site (reviews.json, phrases.json, scoring.db,
                # review_queue.json, the .xlsx, ...)
                evicted = r2_sync.upload_folder(folder, ROOT)
                if evicted:
                    st.warning(
                        f"R2 сховище перевищило ліміт ({r2_sync.MAX_BUCKET_BYTES // 10**9}ГБ) — "
                        f"видалено {evicted} найдавніших файлів (усіх продуктових ліній).")

                llms = [llm_extract, llm_group, llm_verify,
                        llm_reassign, llm_consolidate]
                total_in = sum(l.input_tokens for l in llms)
                total_out = sum(l.output_tokens for l in llms)
                total_cost = sum(l.cost_usd for l in llms)
                total_cost_known = all(l.cost_known for l in llms)
                append_usage_history({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "date": date.today().isoformat(),
                    "product_line": folder.name,
                    "provider": provider,
                    "extract_model": llm_extract.model,
                    "group_model": llm_group.model,
                    "reviews": len(all_reviews),
                    "phrases": len(phrases),
                    "groups": len(tax.groups),
                    "calls": sum(l.calls for l in llms),
                    "cache_hits": sum(l.cache_hits for l in llms),
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "cache_read_tokens": sum(l.cache_read_tokens for l in llms),
                    "cache_write_tokens": sum(l.cache_write_tokens for l in llms),
                    "cost_usd": total_cost,
                    "cost_known": total_cost_known,
                })
                r2_sync.upload_file(ROOT / ROOT_DB_NAME, ROOT)

                st.success(
                    f"**{out.name}** — {len(all_reviews)} відгуків, "
                    f"{len(phrases)} фраз, {len(tax.groups)} груп.")
                with st.container(border=True):
                    st.caption("Екстракція: " + llm_extract.usage_report())
                    st.caption("Групування: " + llm_group.usage_report())
                    if llm_verify.calls:
                        st.caption("Перевірка груп: " + llm_verify.usage_report())
                    if llm_reassign.calls:
                        st.caption("Перепризначення: " + llm_reassign.usage_report())
                    st.caption("Консолідація: " + llm_consolidate.usage_report())
                    cols = st.columns(3)
                    cols[0].metric("Токенів (вхід)", f"{total_in:,}".replace(",", " "))
                    cols[1].metric("Токенів (вихід)", f"{total_out:,}".replace(",", " "))
                    cols[2].metric("Вартість",
                                   f"${total_cost:.4f}" if total_cost_known else "н/д")
                excel_download(out, "dl_run")
            except Exception as e:
                st.error(f"Помилка: {e}")
                st.exception(e)

    # existing workbooks
    old = sorted(folder.glob("*.xlsx"), reverse=True)
    if old:
        st.divider()
        with st.container(border=True):
            st.markdown("**🗂 Готові книги в папці**")
            for i, x in enumerate(old):
                excel_download(x, f"dl_old_{i}")


# ---------------------------------------------------------------- results
with tab_res:
    res_db = taxonomy_db(folder)
    if res_db is None or not res_db.has_taxonomy():
        st.info("Ще немає таксономії — запустіть пайплайн.")
    else:
        tax = res_db.load_taxonomy()
        all_products = sorted({p for c in tax.canonicals.values() for p in c.votes})

        query = st.text_input("🔍 Пошук фрази (по всіх категоріях)",
                              placeholder="bleeding …")
        if query.strip():
            q = normalize(query)
            hits = sorted(
                [c for c in tax.canonicals.values() if q in normalize(c.text)],
                key=lambda c: -c.total)
            st.caption(f"Знайдено: {len(hits)}")
            if hits:
                picked_hit = st.selectbox(
                    "Оберіть фразу, щоб побачити всі варіанти",
                    hits, key="search_pick",
                    format_func=lambda c: (
                        f"{c.text}  ·  {c.total} голосів  —  "
                        f"{CAT_LABELS.get(tax.groups[c.group_id].category, '?')} "
                        f"/ «{tax.groups[c.group_id].name}»"))
                with st.container(border=True):
                    render_phrase_detail(tax, picked_hit, all_products)
            st.divider()

        cats = [c for c in domain_mod.active().ids() if tax.groups_for(c)]
        if not cats:
            st.info("Таксономія порожня.")
        else:
            for cat, tab in zip(cats, st.tabs([cat_label(c) for c in cats])):
                with tab:
                    groups = sorted(tax.groups_for(cat),
                                    key=lambda g: -g.total(tax))
                    total_votes = sum(g.total(tax) for g in groups)
                    st.caption(f"{len(groups)} груп · "
                               f"{sum(len(g.canonicals(tax)) for g in groups)} фраз · "
                               f"{total_votes} голосів")
                    for g in groups:
                        cans = sorted(g.canonicals(tax), key=lambda c: -c.total)
                        title = f"📁 {g.name} — {len(cans)} фраз · **{g.total(tax)}** голосів"
                        if g.usage_category:
                            title = f"[{g.usage_category}] {title}"
                        with st.expander(title, expanded=False):
                            rows = [{"Фраза": c.text, "Разом": c.total,
                                    **{p: c.votes.get(p, 0) for p in all_products}}
                                   for c in cans]
                            df = pd.DataFrame(rows)
                            col_cfg = {"Разом": st.column_config.ProgressColumn(
                                "Разом", format="%d", min_value=0,
                                max_value=max((c.total for c in cans), default=1))}
                            event = st.dataframe(
                                df, hide_index=True, width="stretch",
                                height=min(36 * (len(cans) + 1) + 3, 320),
                                column_config=col_cfg,
                                on_select="rerun", selection_mode="single-row",
                                key=f"phrase_tbl_{g.id}")
                            sel = event.selection.rows if event and event.selection else []
                            st.divider()
                            if sel:
                                with st.container(border=True):
                                    render_phrase_detail(tax, cans[sel[0]], all_products)
                            else:
                                st.caption("👆 Виберіть рядок у таблиці, щоб побачити "
                                          "повні варіанти формулювання цієї фрази.")


# ---------------------------------------------------------------- USP board
# The whole board is one fragment. A filter change, a checkbox tick or
# opening the quotes panel now reruns THIS function only — not all ~2.5k
# lines of the script, which Streamlit otherwise re-executes for every tab,
# hidden ones included (4 load_taxonomy() calls and ~90 dataframes per
# click). Anything that WRITES still calls a bare st.rerun(): that stays
# app-scoped even from inside a fragment, so the other tabs can never show
# a taxonomy the board has already changed.
@st.fragment
def _board_tab(folder: Path) -> None:
    bd_db = taxonomy_db(folder)
    if bd_db is None or not bd_db.has_taxonomy():
        st.info("Ще немає таксономії — запустіть пайплайн на вкладці «Запуск».")
        return
    tax = bd_db.load_taxonomy()
    bd_cats = [c for c in domain_mod.active().ids() if tax.groups_for(c)]
    if not bd_cats:
        st.info("Таксономія порожня.")
    else:
        all_products = sorted({p for c in tax.canonicals.values()
                               for p in c.votes})
        b1, bp, b2, b3, b4 = st.columns([2.4, 2, 1.2, 1.7, 1.5])
        bd_cat = b1.selectbox("Категорія", bd_cats, format_func=cat_label,
                              key="bd_cat")
        ALL_PROD = "— усі товари —"
        bd_prod = bp.selectbox(
            "Товар", [ALL_PROD] + all_products, key="bd_prod",
            help="Показати лише ті фрази, за які голосував цей товар. "
                 "Редагування діють на всю категорію, не лише на нього.")
        bd_prod = None if bd_prod == ALL_PROD else bd_prod
        bd_min = b2.number_input(
            "Мін. голосів", min_value=0, max_value=99, value=0,
            key="bd_min", help="Сховати рідкісні фрази, щоб дошка не рябіла")
        bd_auto = b3.toggle(
            "🤖 Автозлиття схожих", value=True, key="bd_auto",
            help="Після кожного перетягування рядки, які детермінований "
                 "гейт визнає тим самим повідомленням, зливаються самі "
                 "(те саме правило, з яким пайплайн зливає без LLM). "
                 "Решту схожих показано нижче як кандидатів.")
        bd_stack = st.session_state.get(BOARD_UNDO) or []
        if b4.button(f"↩ Скасувати ({len(bd_stack)})", key="bd_undo",
                     width="stretch", disabled=not bd_stack,
                     help=(f"Останнє: {bd_stack[-1][0]}" if bd_stack
                           else "Немає що скасовувати")):
            board_undo(folder, bd_db)
            st.rerun()

        bd_msg = st.session_state.pop("board_msg", None)
        if bd_msg:
            if bd_msg.get("error"):
                st.warning(bd_msg["error"])
            else:
                st.success(bd_msg["note"])
                if bd_msg.get("auto"):
                    st.caption("🤖 Автоматично злито: " +
                               " · ".join(bd_msg["auto"]))
                for line in bd_msg.get("removed", []):
                    st.caption(f"🧹 {line}")
                st.session_state["board_last"] = bd_msg

        # a pending drop waits for the user's choice (which name / which
        # wording survives) before anything is written
        bd_pending = st.session_state.get("board_pending")
        if bd_pending:
            if bd_pending["op"] == "merge_groups":
                board_merge_groups_dialog(folder, bd_db, bd_pending, bd_auto)
            elif bd_pending["op"] == "merge_rows":
                board_merge_rows_dialog(folder, bd_db, bd_pending, bd_auto)
            elif bd_pending["op"] == "new_group":
                board_new_group_dialog(folder, bd_db, bd_pending, bd_auto)

        # raw quotes waiting to be re-homed or thrown out (from the 🔍
        # panel below) — one dialog at a time
        bd_qmove = st.session_state.get("board_quote_move")
        bd_qdrop = st.session_state.get("board_quote_drop")
        if bd_qmove:
            board_move_quotes_dialog(folder, bd_db, bd_qmove)
        elif bd_qdrop:
            board_drop_quotes_dialog(folder, bd_db, bd_qdrop)

        v1, v2, v3 = st.columns([3, 1.7, 1.1])
        bd_view = v1.radio(
            "Вигляд", ["🧩 Дошка (перетягування)", "📋 Результуюча таблиця"],
            horizontal=True, label_visibility="collapsed", key="bd_view")
        bd_height = v2.slider("Висота дошки", 420, 1200, 660, 40,
                              key="bd_h", label_visibility="collapsed",
                              disabled=bd_view.startswith("📋"))
        # the full instructions live in one place, folded away: on screen
        # they were a wall of text between the controls and the board
        with v3.popover("❓ Як це працює", width="stretch"):
            st.markdown(
                "**Найшвидший шлях — меню `⋯` на картці** (або права "
                "кнопка миші): перенести в іншу USP, злити з іншою "
                "фразою, зробити нову USP, показати сирі цитати, "
                "переписати формулювання. Те саме меню є в заголовку "
                "колонки — для дій з усією USP.\n\n"
                "**Перетягування** (те саме, але мишею):\n"
                "- картка **в іншу колонку** — фраза переїде в ту USP;\n"
                "- картка **на іншу картку** — рядки зіллються "
                "(спитаємо, яке формулювання лишити);\n"
                "- **заголовок на заголовок** — зіллються USP "
                "(спитаємо, яку назву лишити);\n"
                "- у зону **➕ Нова USP** — створиться нова група.\n"
                "Біля краю дошки вона прокручується сама — тягнути "
                "можна й у колонку, якої зараз не видно.\n\n"
                "**Масово:** клік по картці (або ☐) виділяє, "
                "**Shift+клік** — цілий діапазон, ☐ у заголовку — усю "
                "колонку. Над дошкою зʼявиться панель: перенести всі "
                "виділені в будь-яку USP, зробити з них нову USP або "
                "злити в один рядок. Виділене можна тягнути гуртом.\n\n"
                "**Сирі цитати** (`⋯ → показати сирі цитати` або 🔍): "
                "☐ біля цитати виділяє її, `☑ Усі` — всі цитати рядка. "
                "Над списком: `↗ Перенести` (в іншу фразу, окремими "
                "рядками або одним новим рядком), `🚫 Без USP` (кожна "
                "цитата стає окремим рядком у групі «Без USP» — вона "
                "лишається в Excel, але вже не роздуває реальну USP), "
                "`🗑 Прибрати` (цитати з їхніми голосами зникають з "
                "аналізу). Усе це записується в overrides.json і "
                "повторюється на наступних прогонах.\n\n"
                "**Usage-група** (лише для категорій зі смугами, напр. "
                "«Usage»): чіп `[…]` під назвою колонки — це смуга, під "
                "якою USP стане в Excel. Клік по чіпу — вибрати наявну "
                "смугу, створити нову, перейменувати смугу в усіх USP "
                "одразу або зняти її. USP без смуги показано як "
                "`[+ usage-група]` і в Excel вона піде у смугу за "
                "замовчуванням.\n\n"
                "**Клавіші:** `/` — пошук, `Esc` — зняти виділення / "
                "закрити меню, `Enter` — підтвердити перейменування, "
                "подвійний клік по тексту — переписати його.\n\n"
                "**Колонки** згортаються (`⇤`), щоб довга дошка "
                "вміщалася; згорнута колонка лишається місцем, куди "
                "можна кинути фразу.")

        if bd_view.startswith("🧩"):
            st.caption(
                "Тягніть картки або тисніть **⋯** на картці — усі дії "
                "доступні з меню. `/` — пошук, `Esc` — зняти виділення."
                + (f" · Фільтр товару: **{bd_prod}** (показано лише його "
                   "фрази)" if bd_prod else ""))
            bd_ins = st.session_state.get("board_inspect")
            if bd_ins and bd_ins not in tax.canonicals:
                bd_ins = None                       # merged away meanwhile
                st.session_state.pop("board_inspect", None)
            bd_op = taxonomy_board(
                board_payload(tax, bd_cat, min_votes=bd_min,
                              product=bd_prod, inspect=bd_ins),
                height=bd_height,
                key=f"board_{folder.name}_{bd_cat}")
            # Streamlit replays the last component value on every rerun —
            # act once per nonce, else the drop repeats forever
            bd_nonce_key = f"bd_nonce_{folder.name}_{bd_cat}"
            if bd_op and st.session_state.get(bd_nonce_key) != bd_op.get("nonce"):
                st.session_state[bd_nonce_key] = bd_op.get("nonce")
                kind = bd_op.get("op")
                if kind == "move_rows":
                    board_edit(
                        folder, bd_db, "перенесення фраз",
                        lambda tax, ov: (
                            me.move_rows(tax, ov, bd_op["rows"],
                                         bd_op["to_group"]),
                            [bd_op["to_group"]], list(bd_op["rows"])),
                        bd_auto)
                    st.rerun()
                elif kind == "rename_group":
                    board_edit(
                        folder, bd_db, "перейменування USP",
                        lambda tax, ov: (
                            me.rename_group(tax, ov, bd_op["group"],
                                            bd_op["name"]),
                            [bd_op["group"]], []),
                        False)
                    st.rerun()
                elif kind == "rename_row":
                    board_edit(
                        folder, bd_db, "перейменування фрази",
                        lambda tax, ov: (
                            me.rename_row(tax, ov, bd_op["row"],
                                          bd_op["text"]),
                            [], []),
                        False)
                    st.rerun()
                elif kind == "set_usage_category":
                    board_edit(
                        folder, bd_db, "зміну usage-групи",
                        lambda tax, ov: (
                            me.set_usage_category(tax, ov, bd_op["group"],
                                                  bd_op.get("bucket", "")),
                            [bd_op["group"]], []),
                        False)
                    st.rerun()
                elif kind == "rename_usage_bucket":
                    def _rename_band(tax, ov):
                        note, gids = me.rename_usage_bucket(
                            tax, ov, bd_cat, bd_op.get("old", ""),
                            bd_op.get("name", ""))
                        return note, gids, []
                    board_edit(folder, bd_db, "перейменування usage-групи",
                               _rename_band, False)
                    st.rerun()
                elif kind == "inspect":
                    # read-only: nothing is written, no undo entry
                    st.session_state["board_inspect"] = bd_op["row"]
                    st.rerun(scope="fragment")
                elif kind in ("merge_groups", "merge_rows", "new_group"):
                    st.session_state["board_pending"] = bd_op
                    st.rerun()
            if bd_ins and bd_ins in tax.canonicals:
                board_quotes_panel(folder, bd_db, bd_cat,
                                   tax.canonicals[bd_ins])
        else:
            bd_rows = []
            for g in sorted(tax.groups_for(bd_cat),
                            key=lambda g: -g.total(tax)):
                for c in sorted(g.canonicals(tax), key=lambda c: -c.total):
                    # same filters as the board, so the table always shows
                    # what the board shows — just laid out as Excel will
                    shown = c.votes.get(bd_prod, 0) if bd_prod else c.total
                    if shown < (max(bd_min, 1) if bd_prod else bd_min):
                        continue
                    bd_rows.append({
                        "USP": (f"[{g.usage_category}] {g.name}"
                                if g.usage_category else g.name),
                        "Фраза": c.text, "Разом": c.total,
                        **{p: c.votes.get(p, 0) for p in all_products}})
            bd_df = pd.DataFrame(bd_rows)
            st.caption(f"{len(tax.groups_for(bd_cat))} USP · "
                       f"{len(bd_rows)} рядків — так вони ляжуть в Excel.")
            st.dataframe(
                bd_df, hide_index=True, width="stretch", height=560,
                column_config={"Разом": st.column_config.ProgressColumn(
                    "Разом", format="%d", min_value=0,
                    max_value=int(bd_df["Разом"].max()) if len(bd_df) else 1)})
            st.download_button(
                "📥 CSV цієї таблиці",
                data=bd_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{folder.name}-{bd_cat}.csv", mime="text/csv",
                key="bd_csv")

        # ---- usage bands: how this category will be banded in Excel.
        # The board sorts columns by votes, so the band a USP belongs to is
        # only visible per-column — this panel is where you see the whole
        # banding at once, and the USPs that have no band at all.
        if domain_mod.active().has_subbucket(bd_cat):
            st.divider()
            bd_bands: dict[str, list] = {}
            for g in sorted(tax.groups_for(bd_cat),
                            key=lambda g: -g.total(tax)):
                bd_bands.setdefault(g.usage_category, []).append(g)
            bd_orphans = bd_bands.pop("", [])
            with st.expander(
                    f"🏷 Usage-групи ({len(bd_bands)} смуг · "
                    f"{len(bd_orphans)} USP без смуги)",
                    expanded=bool(bd_orphans)):
                st.caption("Смуга — заголовок над групою USP на аркуші "
                           "Excel. Змінити смугу однієї USP можна й на "
                           "дошці — чіп `[…]` у заголовку колонки.")
                for band, gs in sorted(
                        bd_bands.items(),
                        key=lambda kv: -sum(g.total(tax) for g in kv[1])):
                    st.markdown(
                        f"**{band}** — {len(gs)} USP · "
                        f"{sum(g.total(tax) for g in gs)} гол.")
                    st.caption(" · ".join(f"{g.name} ({g.total(tax)})"
                                          for g in gs))
                if bd_orphans:
                    st.markdown(f"**— без смуги — ({len(bd_orphans)} USP)**")
                    st.caption("В Excel вони підуть у смугу за "
                               "замовчуванням. Призначте смугу тут:")
                    bd_opts = me.usage_buckets(tax, bd_cat)
                    BAND_NEW = "✏️ нова смуга…"
                    for g in bd_orphans:
                        o1, o2, o3 = st.columns([4, 4, 1])
                        o1.markdown(f"«{g.name}» · {g.total(tax)} гол.")
                        pick = o2.selectbox(
                            "Смуга", bd_opts + [BAND_NEW],
                            key=f"bd_band_{g.id}",
                            label_visibility="collapsed")
                        val = pick
                        if pick == BAND_NEW:
                            val = o2.text_input(
                                "Назва смуги", key=f"bd_bandnew_{g.id}",
                                label_visibility="collapsed",
                                placeholder="назва нової смуги")
                        val = (val or "").strip()
                        if o3.button("🏷", key=f"bd_bandset_{g.id}",
                                     help="Призначити цю смугу",
                                     disabled=not val):
                            board_edit(
                                folder, bd_db, "зміну usage-групи",
                                lambda tax, ov, gid=g.id, b=val: (
                                    me.set_usage_category(tax, ov, gid, b),
                                    [gid], []),
                                False)
                            st.rerun()

        # ---- what else could be merged (deterministic, no LLM calls)
        st.divider()
        bd_last = st.session_state.get("board_last") or {}
        bd_moved = [r for r in bd_last.get("rows", []) if r in tax.canonicals]
        bd_moved = [r for r in bd_moved
                    if me.category_of(tax, tax.canonicals[r]) == bd_cat]
        if bd_moved:
            bd_follow = me.suggest_row_moves(tax, bd_moved)
            if bd_follow:
                tgt_gid = tax.canonicals[bd_moved[0]].group_id
                tgt_name = tax.groups[tgt_gid].name
                with st.container(border=True):
                    st.markdown(f"**🧲 Схожі фрази в інших USP** — "
                                f"перенести теж у «{tgt_name}»?")
                    for i, s in enumerate(bd_follow):
                        f1, f2 = st.columns([12, 1])
                        f1.markdown(f"«{s['text']}» · **{s['score']}** "
                                    f"схожості з «{s['like']}» — зараз у "
                                    f"«{s['from']}»")
                        if f2.button("➡", key=f"bd_follow_{i}",
                                     help="Перенести сюди"):
                            board_edit(
                                folder, bd_db, "перенесення схожої фрази",
                                lambda tax, ov, rid=s["row"]: (
                                    me.move_rows(tax, ov, [rid], tgt_gid),
                                    [tgt_gid], [rid]),
                                bd_auto)
                            st.rerun()
                    if len(bd_follow) > 1 and st.button(
                            "➡ Перенести всі", key="bd_follow_all"):
                        ids = [s["row"] for s in bd_follow]
                        board_edit(
                            folder, bd_db, "перенесення схожих фраз",
                            lambda tax, ov: (
                                me.move_rows(tax, ov, ids, tgt_gid),
                                [tgt_gid], ids),
                            bd_auto)
                        st.rerun()

        bd_dups: list[dict] = []
        for g in tax.groups_for(bd_cat):
            bd_dups += me.suggest_row_merges(tax, g.id, limit=4)
        bd_dups.sort(key=lambda d: -d["score"])
        bd_dups = bd_dups[:10]
        bd_gpairs = me.suggest_group_merges(tax, bd_cat)
        if bd_dups or bd_gpairs:
            with st.expander(
                    f"🤖 Кандидати на об'єднання "
                    f"({len(bd_dups)} рядків · {len(bd_gpairs)} USP)",
                    expanded=bool(bd_moved)):
                st.caption("Лексична схожість, без LLM. Те, що гейт "
                           "визнає тим самим повідомленням, уже злито "
                           "автоматично — тут лишилось те, що потребує "
                           "вашого рішення.")
                for i, d in enumerate(bd_dups):
                    d1, d2 = st.columns([12, 1])
                    why = (f" · ⛔ {d['blocked']}" if d["blocked"]
                           else " · збіг за словами")
                    d1.markdown(f"«{d['other_text']}» → «{d['keep_text']}» "
                                f"· **{d['score']}**{why}")
                    if d2.button("🔗", key=f"bd_dup_{i}",
                                 help="Злити в популярніший рядок"):
                        st.session_state["board_pending"] = {
                            "op": "merge_rows", "rows": [d["other"]],
                            "into": d["keep"], "nonce": 0}
                        st.rerun()
                if bd_dups and bd_gpairs:
                    st.divider()
                for i, p in enumerate(bd_gpairs):
                    g1, g2 = st.columns([12, 1])
                    g1.markdown(f"USP «{p['other_name']}» → "
                                f"«{p['keep_name']}» · **{p['score']}**")
                    if g2.button("🔀", key=f"bd_gp_{i}",
                                 help="Злити ці USP"):
                        st.session_state["board_pending"] = {
                            "op": "merge_groups", "source": p["other"],
                            "target": p["keep"], "nonce": 0}
                        st.rerun()

        st.divider()
        bd_c1, bd_c2 = st.columns([2, 3])
        if bd_c1.button("📄 Перегенерувати Excel", type="primary",
                        width="stretch", key="bd_excel"):
            out = regenerate_excel(folder)
            st.toast(f"Записано {out.name} (без LLM-викликів)", icon="📄")
            excel_download(out, "dl_board")
        bd_c2.caption("Кожна дія тут одразу записується в scoring.db **і** "
                      "в overrides.json — тож переживає наступні прогони "
                      "пайплайна (навіть «з нуля»). Список правил — на "
                      "вкладці «✏️ Корекції».")


with tab_board:
    _board_tab(folder)


# ---------------------------------------------------------------- per-product
with tab_by_prod:
    bp_db = taxonomy_db(folder)
    if bp_db is None or not bp_db.has_taxonomy():
        st.info("Ще немає таксономії — запустіть пайплайн.")
    else:
        tax = bp_db.load_taxonomy()
        all_products = sorted({p for c in tax.canonicals.values() for p in c.votes})
        if not all_products:
            st.info("Таксономія порожня.")
        else:
            phrase_counts = load_phrase_counts(folder)

            f1, f2, f3 = st.columns([2, 2, 1])
            pv_prod = f1.selectbox("Товар", all_products, key="pv_prod")
            pv_cats = [c for c in domain_mod.active().ids()
                       if any(cn.votes.get(pv_prod, 0)
                              for g in tax.groups_for(c)
                              for cn in g.canonicals(tax))]
            pv_cat = f2.selectbox("Категорія", pv_cats,
                                  format_func=lambda c: CAT_LABELS.get(c, c),
                                  key="pv_cat")
            pv_min = f3.number_input(
                "Мін. голосів", min_value=1, value=3, key="pv_min",
                help="3 = лише фрази, де більше двох голосів у цього товару")

            pv_groups = []
            for g in tax.groups_for(pv_cat):
                cans = [c for c in g.canonicals(tax)
                        if c.votes.get(pv_prod, 0) >= pv_min]
                if cans:
                    cans.sort(key=lambda c: -c.votes.get(pv_prod, 0))
                    pv_groups.append((g, cans))
            pv_groups.sort(
                key=lambda gc: -sum(c.votes.get(pv_prod, 0) for c in gc[1]))

            if not pv_groups:
                st.warning(f"У товару «{pv_prod}» немає фраз із {pv_min}+ "
                           f"голосами в цій категорії — зменшіть поріг.")
            else:
                n_phr = sum(len(cans) for _, cans in pv_groups)
                st.caption(f"**{pv_prod}** · груп: **{len(pv_groups)}** · "
                           f"фраз із {pv_min}+ голосами: **{n_phr}**. "
                           "Під кожною фразою — сирі цитати з відгуків, "
                           "які було злито в неї (⭐ = збігається з канонічною, "
                           "×N = скільки відгуків так сказали).")
                for g, cans in pv_groups:
                    g_votes = sum(c.votes.get(pv_prod, 0) for c in cans)
                    title = f"📁 {g.name} — {len(cans)} фраз · **{g_votes}** голосів"
                    if g.usage_category:
                        title = f"[{g.usage_category}] {title}"
                    with st.expander(title, expanded=len(pv_groups) <= 3):
                        for k, c in enumerate(cans):
                            if k:
                                st.divider()
                            votes = c.votes.get(pv_prod, 0)
                            st.markdown(f"**💬 «{html.escape(c.text)}»** — "
                                        f"**{votes}** голос(ів)")
                            variants = split_quote_variants(
                                c.quotes.get(pv_prod, []))
                            covered = 0
                            for v in variants:
                                n = phrase_counts.get(
                                    (pv_cat, pv_prod, normalize(v)), 0)
                                covered += n or 1
                                mark = (" ⭐" if normalize(v) == normalize(c.text)
                                        else "")
                                st.markdown(
                                    f'<div class="quote-card">'
                                    f'<div class="quote-product">×{n or "?"}{mark}'
                                    f'</div>{html.escape(v)}</div>',
                                    unsafe_allow_html=True)
                            if not variants:
                                st.caption("Сирі цитати для цього товару "
                                           "не збереглися.")
                            elif covered < votes:
                                st.caption(f"⚠️ Показані цитати покривають "
                                           f"{covered} з {votes} голосів — "
                                           f"решта варіантів не збереглася "
                                           f"(зберігається до 3 на злиття).")


# ---------------------------------------------------------------- review queue
with tab_review:
    rq_path = folder / "review_queue.json"
    if not rq_path.exists():
        st.info("Черга перевірки з'явиться після наступного прогону пайплайна.")
    else:
        rq = json.loads(rq_path.read_text(encoding="utf-8"))
        if not rq:
            st.success("Останній прогін не залишив сумнівних місць 🎉")
        else:
            st.caption("Місця, де пайплайн вагався. Перегляньте і за потреби "
                       "виправте на вкладці «✏️ Корекції» — правила "
                       "переживають повторні прогони.")
            RQ_SECTIONS = [
                ("auto_merge", "🧲 Автоматичні злиття без судді",
                 "Формулювання, які детермінований гейт визнав тим самим "
                 "повідомленням (одна й та сама похвала, інший підмет/"
                 "одруківка) і злив без LLM. Якщо якесь злиття хибне — "
                 "«Корекції → Перейменувати/Перенести»."),
                ("row_merge", "🪢 Злиті рядки-синоніми (LLM-пас)",
                 "Рядки однієї групи, які суддя визнав тим самим "
                 "повідомленням (don't stick == won't hold == no adhesion) "
                 "і злив у найпопулярніше формулювання. Якщо якесь злиття "
                 "хибне — «Корекції → Перейменувати/Злити фрази» навпаки."),
                ("gate_blocked", "🚧 Розділені гейтом злиття",
                 "Суддя хотів злити фразу в існуючий рядок, але це порушує "
                 "жорстке правило (негація з одного боку, різні "
                 "кваліфікатори, різні рівні похвали, довге речення проти "
                 "короткого рядка) — фраза стала окремим рядком у тій самій "
                 "групі. Якщо це насправді одне й те саме — «Корекції → "
                 "Злити фрази»."),
                ("gate_overridden", "🧠 Зняті вето (за вашими прецедентами)",
                 "Гейт хотів заблокувати ці злиття, але ваші попередні "
                 "вердикти зняли вето: точний прецедент ✗, схожий прецедент "
                 "або ослаблене вагами правило. Перевірте: ✓ — злиття "
                 "правильне (підкріплює прецедент), ✗ — хибне (наступний "
                 "прогін залишить фрази окремими рядками)."),
                ("row_dissolved", "🫧 Рядки, що розчинилися при перепризначенні",
                 "Усі варіанти цього рядка при фінальному пасі розійшлися по "
                 "інших рядках/групах."),
                ("consolidate", "🧹 Дії консолідації",
                 "Злиття груп-дублікатів і переноси рядків фінальним аудитором."),
                ("singleton_group", "🧍 Групи з одним рядком і ≤2 голосами",
                 "Можливо, це переформульований дублікат існуючої теми — "
                 "«Корекції → Злити групи»."),
                ("fallback", "🆘 Аварійні розміщення",
                 "Суддя пропустив фразу — її покладено поруч із лексично "
                 "найближчим рядком. Перевірте групу."),
            ]
            cat_lbl = lambda c: CAT_LABELS.get(c, c) if c else ""
            gate_labels = load_gate_labels(folder)
            for key, title, hint in RQ_SECTIONS:
                items = [a for a in rq if a.get("type") == key]
                if not items:
                    continue

                if key == "gate_blocked":
                    n_done = sum(1 for a in items
                                 if gate_label_key(a) in gate_labels)
                    with st.expander(f"{title} ({len(items)}, "
                                     f"розмічено {n_done})", expanded=True):
                        st.caption(hint)
                        st.caption("**✓** — вето правильне, фрази лишаються "
                                   "окремими рядками. **✗** — вето хибне, "
                                   "злити: правило одразу додається у "
                                   "«Корекції → Злити фрази». Позначки "
                                   "зберігаються в gate_labels.json і з "
                                   "наступного прогону діють як прецеденти: "
                                   "той самий (або дуже схожий) випадок "
                                   "вирішується вашим вердиктом автоматично "
                                   "(лише для цього товару), а правило, чиї "
                                   "вето часто хибні, ослаблюється вагами — "
                                   "спільними для ВСІХ товарів (файл "
                                   "products/gate_rule_weights.json).")
                        shared = aggregate_rule_weights(folder.parent)
                        gp = GatePrecedents(gate_labels,
                                            cfg.get("gate_feedback"), shared)
                        eff = gp.effective_stats()
                        if eff:
                            wl = []
                            for reason, (kn, mn) in sorted(eff.items()):
                                _, _, soft = gp.rule_softness(reason)
                                state = ("⚖️ ослаблене — вето знімається"
                                         if soft else "тримається")
                                wl.append(f"«{reason}»: ✓{kn} / ✗{mn} — "
                                          f"{state}")
                            st.caption("**Ваги правил гейта (всі товари):** " +
                                       " · ".join(wl))
                        hide_done = st.toggle("Сховати розмічені", value=True,
                                              key="gate_hide_done")
                        for i, a in enumerate(items):
                            gk = gate_label_key(a)
                            lab = gate_labels.get(gk)
                            if lab and hide_done:
                                continue
                            why = (f" — {a['reason']}"
                                   if a.get("reason") else "")
                            if a.get("basis"):
                                why += f" · {a['basis']}"
                            line = (f"«{a['phrase']}» ≠ «{a['into']}»{why} "
                                    f"(група **{a['group']}**, "
                                    f"{cat_lbl(a['category'])})")
                            col_t, col_y, col_n = st.columns([12, 1, 1])
                            if lab is None:
                                col_t.markdown(line)
                                if col_y.button(
                                        "✓", key=f"gate_y_{i}",
                                        help="Вето правильне — лишити "
                                             "окремими рядками"):
                                    gate_labels[gk] = {
                                        **{f: a.get(f, "") for f in
                                           ("category", "group", "phrase",
                                            "into", "reason")},
                                        "label": "keep",
                                        "ts": datetime.now().isoformat(
                                            timespec="seconds")}
                                    save_gate_labels(folder, gate_labels)
                                    st.rerun()
                                if col_n.button(
                                        "✗", key=f"gate_n_{i}",
                                        help="Вето хибне — злити фрази "
                                             "(додасться корекція)"):
                                    gate_labels[gk] = {
                                        **{f: a.get(f, "") for f in
                                           ("category", "group", "phrase",
                                            "into", "reason")},
                                        "label": "merge",
                                        "ts": datetime.now().isoformat(
                                            timespec="seconds")}
                                    save_gate_labels(folder, gate_labels)
                                    ov = load_overrides(folder)
                                    rules = ov.setdefault(
                                        "merge_canonicals", [])
                                    keep_n = normalize(a["into"])
                                    phr_n = normalize(a["phrase"])
                                    if not any(
                                            len(r) >= 2
                                            and normalize(r[0]) == keep_n
                                            and phr_n in {normalize(t)
                                                          for t in r[1:]}
                                            for r in rules):
                                        rules.append([a["into"], a["phrase"]])
                                        save_overrides(folder, ov)
                                    st.rerun()
                            else:
                                mark = ("🔗 злито" if lab["label"] == "merge"
                                        else "✓ окремо")
                                col_t.markdown(f"{mark} · {line}")
                                if col_y.button("↩", key=f"gate_u_{i}",
                                                help="Скасувати позначку"):
                                    gate_labels.pop(gk, None)
                                    save_gate_labels(folder, gate_labels)
                                    if lab["label"] == "merge":
                                        ov = load_overrides(folder)
                                        rules = ov.get("merge_canonicals", [])
                                        target = [normalize(a["into"]),
                                                  normalize(a["phrase"])]
                                        for r in list(rules):
                                            if [normalize(t)
                                                    for t in r] == target:
                                                rules.remove(r)
                                        save_overrides(folder, ov)
                                    st.rerun()
                        if n_done:
                            st.caption("🔗-позначки вже видно у «Корекції» — "
                                       "застосуйте їх там кнопкою «Застосувати "
                                       "й перегенерувати Excel».")
                    continue

                if key == "gate_overridden":
                    n_done = sum(1 for a in items
                                 if gate_label_key(a) in gate_labels)
                    with st.expander(f"{title} ({len(items)}, "
                                     f"розмічено {n_done})", expanded=True):
                        st.caption(hint)
                        hide_done_o = st.toggle("Сховати розмічені",
                                                value=True,
                                                key="ovr_hide_done")
                        for i, a in enumerate(items):
                            gk = gate_label_key(a)
                            lab = gate_labels.get(gk)
                            if lab and hide_done_o:
                                continue
                            line = (f"«{a['phrase']}» 🔗 «{a['into']}» — "
                                    f"{a.get('basis', '')} "
                                    f"(група **{a['group']}**, "
                                    f"{cat_lbl(a['category'])})")
                            col_t, col_y, col_n = st.columns([12, 1, 1])
                            if lab is None:
                                col_t.markdown(line)
                                if col_y.button(
                                        "✓", key=f"ovr_y_{i}",
                                        help="Злиття правильне — "
                                             "підкріпити прецедент"):
                                    gate_labels[gk] = {
                                        **{f: a.get(f, "") for f in
                                           ("category", "group", "phrase",
                                            "into", "reason")},
                                        "label": "merge",
                                        "ts": datetime.now().isoformat(
                                            timespec="seconds")}
                                    save_gate_labels(folder, gate_labels)
                                    st.rerun()
                                if col_n.button(
                                        "✗", key=f"ovr_n_{i}",
                                        help="Злиття хибне — наступний "
                                             "прогін залишить фрази "
                                             "окремими рядками"):
                                    gate_labels[gk] = {
                                        **{f: a.get(f, "") for f in
                                           ("category", "group", "phrase",
                                            "into", "reason")},
                                        "label": "keep",
                                        "ts": datetime.now().isoformat(
                                            timespec="seconds")}
                                    save_gate_labels(folder, gate_labels)
                                    # прибрати ручне правило злиття, якщо
                                    # воно було створено раніше кнопкою ✗
                                    ov = load_overrides(folder)
                                    rules = ov.get("merge_canonicals", [])
                                    target = [normalize(a["into"]),
                                              normalize(a["phrase"])]
                                    for r in list(rules):
                                        if [normalize(t) for t in r] == target:
                                            rules.remove(r)
                                    save_overrides(folder, ov)
                                    st.rerun()
                            else:
                                mark = ("✓ злито" if lab["label"] == "merge"
                                        else "✂️ розділити")
                                col_t.markdown(f"{mark} · {line}")
                                if col_y.button("↩", key=f"ovr_u_{i}",
                                                help="Скасувати позначку"):
                                    gate_labels.pop(gk, None)
                                    save_gate_labels(folder, gate_labels)
                                    st.rerun()
                    continue

                with st.expander(f"{title} ({len(items)})", expanded=(
                        key == "row_dissolved")):
                    st.caption(hint)
                    for a in items:
                        if key in ("auto_merge", "row_merge"):
                            st.markdown(f"- «{a['phrase']}» → «{a['into']}» "
                                        f"(група **{a['group']}**, "
                                        f"{cat_lbl(a['category'])})")
                        elif key == "row_dissolved":
                            st.markdown(f"- «{a['row']}» з групи "
                                        f"**{a['group']}** "
                                        f"({cat_lbl(a['category'])})")
                        elif key == "consolidate":
                            st.markdown(f"- {a['action']}")
                        elif key == "singleton_group":
                            st.markdown(f"- **{a['group']}**: «{a['row']}» "
                                        f"({a['votes']} гол., "
                                        f"{cat_lbl(a['category'])})")
                        elif key == "fallback":
                            st.markdown(f"- «{a['phrase']}» → група "
                                        f"**{a['group']}** "
                                        f"({cat_lbl(a['category'])})")


# ---------------------------------------------------------------- overrides
with tab_fix:
    fix_db = taxonomy_db(folder)
    if fix_db is None or not fix_db.has_taxonomy():
        st.info("Корекції доступні після першого прогону пайплайна.")
    else:
        tax = fix_db.load_taxonomy()
        ov = load_overrides(folder)
        group_names = {g.id: g.name for g in tax.groups.values()}

        def glabel(g):
            return f"{g.name}  ·  {CAT_LABELS.get(g.category, g.category)} ({g.total(tax)})"

        groups_sorted = sorted(tax.groups.values(), key=lambda g: -g.total(tax))
        cans_sorted = sorted(tax.canonicals.values(), key=lambda c: -c.total)

        fix_rn, fix_rc, fix_mg, fix_mv, fix_mc, fix_uc = st.tabs(
            ["✏️ Перейменувати групу", "🏷 Перейменувати фразу", "🔀 Злити групи",
             "📌 Перенести фразу", "🔗 Злити фрази", "🏷 Usage-група"])

        with fix_rn:
            with st.container(border=True):
                g_old = st.selectbox("Група", groups_sorted, format_func=glabel,
                                     key="rn_g")
                g_new = st.text_input("Нова назва", key="rn_new")
                if st.button("➕ Додати правило", key="rn_add") and g_new.strip():
                    ov.setdefault("rename", {})[g_old.name] = g_new.strip()
                    save_overrides(folder, ov)
                    st.rerun()

        with fix_rc:
            with st.container(border=True):
                c_old = st.selectbox(
                    "Фраза", cans_sorted, key="rc_c",
                    format_func=lambda c: f"{c.text} ({c.total}) ← "
                                          f"{group_names.get(c.group_id, '?')}")
                c_new = st.text_input("Новий текст рядка", key="rc_new")
                if st.button("➕ Додати правило", key="rc_add") and c_new.strip():
                    ov.setdefault("rename_canonical", {})[c_old.text] = c_new.strip()
                    save_overrides(folder, ov)
                    st.rerun()

        with fix_mg:
            with st.container(border=True):
                keep = st.selectbox("Залишити", groups_sorted, format_func=glabel,
                                    key="mg_keep")
                others = st.multiselect(
                    "Влити в неї", [g for g in groups_sorted
                                    if g.id != keep.id and g.category == keep.category],
                    format_func=glabel, key="mg_others")
                if st.button("➕ Додати правило", key="mg_add") and others:
                    ov.setdefault("merge_groups", []).append(
                        [keep.name] + [g.name for g in others])
                    save_overrides(folder, ov)
                    st.rerun()

        with fix_mv:
            with st.container(border=True):
                can = st.selectbox(
                    "Фраза", cans_sorted, key="mv_c",
                    format_func=lambda c: f"{c.text} ({c.total}) ← "
                                          f"{group_names.get(c.group_id, '?')}")
                src = tax.groups.get(can.group_id)
                targets = [g for g in groups_sorted
                           if g.id != can.group_id
                           and (src is None or g.category == src.category)]
                tgt = st.selectbox("У групу", targets, format_func=glabel, key="mv_g")
                if st.button("➕ Додати правило", key="mv_add"):
                    ov.setdefault("move_canonical", {})[can.text] = tgt.name
                    save_overrides(folder, ov)
                    st.rerun()

        with fix_mc:
            with st.container(border=True):
                can_keep = st.selectbox(
                    "Залишити", cans_sorted, key="mc_keep",
                    format_func=lambda c: f"{c.text} ({c.total}) ← "
                                          f"{group_names.get(c.group_id, '?')}")
                can_others = st.multiselect(
                    "Влити в неї", [c for c in cans_sorted if c.id != can_keep.id],
                    format_func=lambda c: f"{c.text} ({c.total}) ← "
                                          f"{group_names.get(c.group_id, '?')}",
                    key="mc_others")
                if st.button("➕ Додати правило", key="mc_add") and can_others:
                    ov.setdefault("merge_canonicals", []).append(
                        [can_keep.text] + [c.text for c in can_others])
                    save_overrides(folder, ov)
                    st.rerun()

        with fix_uc:
            with st.container(border=True):
                # only categories that actually band their groups (subbucket) —
                # elsewhere the field is written but never read by Excel
                uc_cats = [c for c in domain_mod.active().ids()
                           if domain_mod.active().has_subbucket(c)
                           and tax.groups_for(c)]
                if not uc_cats:
                    st.caption("У цьому домені немає категорій зі смугами "
                               "(usage-групами).")
                else:
                    uc_groups = [g for c in uc_cats for g in groups_sorted
                                 if g.category == c]
                    uc_g = st.selectbox(
                        "USP", uc_groups, key="uc_g",
                        format_func=lambda g: (
                            f"{g.name}  ·  "
                            f"[{g.usage_category or '— без смуги —'}]  "
                            f"({g.total(tax)})"))
                    uc_opts = me.usage_buckets(tax, uc_g.category)
                    UC_NEW = "✏️ нова смуга…"
                    UC_NONE = "🚫 без смуги"
                    uc_pick = st.selectbox("Смуга (usage-група)",
                                           uc_opts + [UC_NEW, UC_NONE],
                                           key="uc_pick")
                    uc_val = "" if uc_pick == UC_NONE else uc_pick
                    if uc_pick == UC_NEW:
                        uc_val = st.text_input("Назва нової смуги",
                                               key="uc_new")
                    uc_val = (uc_val or "").strip()
                    st.caption("Правило застосовується ОСТАННІМ — після "
                               "перейменувань, злиття і створення USP, тож "
                               "переживає їх усі.")
                    if st.button("➕ Додати правило", key="uc_add",
                                 disabled=uc_pick == UC_NEW and not uc_val):
                        ov.setdefault("usage_category", {})[uc_g.name] = uc_val
                        save_overrides(folder, ov)
                        st.rerun()

        st.divider()
        n_rules = sum(len(ov.get(k, ())) for k in
                     ("rename", "rename_canonical", "merge_groups",
                      "move_canonical", "merge_canonicals", "usage_category"))
        st.markdown(f"### 📋 Поточні корекції ({n_rules})")
        if not n_rules:
            st.caption("Поки що порожньо. Корекції застосовуються після кожного "
                       "прогону, тож переживають додавання нових продуктів.")
        else:
            def rule_row(ok: bool, text: str, del_key: str, on_delete):
                cc1, cc2 = st.columns([12, 1])
                cc1.markdown(("✅ " if ok else "⚠️ ") + text +
                             ("" if ok else "  — *групу/фразу не знайдено, "
                                            "правило буде пропущено*"))
                if cc2.button("🗑", key=del_key):
                    on_delete()
                    save_overrides(folder, ov)
                    st.rerun()

            with st.container(border=True):
                for old in list(ov.get("rename", {})):
                    new = ov["rename"][old]
                    rule_row(group_exists(tax, old),
                             f"Перейменувати «{old}» → «{new}»",
                             f"del_rn_{old}",
                             lambda o=old: ov["rename"].pop(o))

                for old in list(ov.get("rename_canonical", {})):
                    new = ov["rename_canonical"][old]
                    rule_row(canonical_exists(tax, old),
                             f"Перейменувати фразу «{old}» → «{new}»",
                             f"del_rc_{old}",
                             lambda o=old: ov["rename_canonical"].pop(o))

                for i, names in enumerate(list(ov.get("merge_groups", []))):
                    ok = all(group_exists(tax, n) for n in names)
                    rule_row(ok,
                             f"Злити {', '.join('«'+n+'»' for n in names[1:])} "
                             f"→ «{names[0]}»",
                             f"del_mg_{i}",
                             lambda idx=i: ov["merge_groups"].pop(idx))

                for text in list(ov.get("move_canonical", {})):
                    tgt_name = ov["move_canonical"][text]
                    ok = canonical_exists(tax, text) and group_exists(tax, tgt_name)
                    rule_row(ok,
                             f"Перенести «{text}» у групу «{tgt_name}»",
                             f"del_mv_{text}",
                             lambda t=text: ov["move_canonical"].pop(t))

                for i, texts in enumerate(list(ov.get("merge_canonicals", []))):
                    ok = all(canonical_exists(tax, t) for t in texts)
                    rule_row(ok,
                             f"Злити {', '.join('«'+t+'»' for t in texts[1:])} "
                             f"→ «{texts[0]}»",
                             f"del_mc_{i}",
                             lambda idx=i: ov["merge_canonicals"].pop(idx))

                for name in list(ov.get("usage_category", {})):
                    band = ov["usage_category"][name]
                    rule_row(group_exists(tax, name),
                             (f"Usage-група «{name}» → «{band}»" if band
                              else f"Зняти usage-групу з «{name}»"),
                             f"del_uc_{name}",
                             lambda n=name: ov["usage_category"].pop(n))

            st.divider()
            if st.button("📄 Застосувати й перегенерувати Excel", type="primary",
                        width="stretch"):
                out = regenerate_excel(folder)
                st.toast(f"Записано {out.name} (без LLM-викликів)", icon="📄")
                excel_download(out, "dl_fix")


# ---------------------------------------------------------------- costs
with tab_cost:
    hist = load_usage_history()
    if hist.empty:
        st.info("Ще немає жодного зафіксованого прогону — витрати з'являться "
                "тут після першого запуску на вкладці «Запуск».")
    else:
        lines = sorted(hist["product_line"].unique())
        f1, f2 = st.columns([2, 1])
        picked_lines = f1.multiselect("Продуктові лінійки", lines, default=lines)
        only_known = f2.checkbox("Лише з відомою ціною", value=False)

        view = hist[hist["product_line"].isin(picked_lines)]
        if only_known:
            view = view[view["cost_known"]]

        if view.empty:
            st.warning("Немає прогонів під обрані фільтри.")
        else:
            known_cost = view.loc[view["cost_known"], "cost_usd"].sum()
            unknown_n = int((~view["cost_known"]).sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Прогонів", len(view))
            m2.metric("Токенів усього",
                      f"{(view['input_tokens'] + view['output_tokens']).sum():,}"
                      .replace(",", " "))
            m3.metric("Витрачено (відомо)", f"${known_cost:.4f}")
            m4.metric("Без відомої ціни", unknown_n if unknown_n else "—")

            daily = (view.groupby(view["date"].dt.date)
                         .agg(cost_usd=("cost_usd", "sum"),
                              tokens=("input_tokens", "sum"))
                         .reset_index())
            if len(daily) >= 2:
                chart = (
                    alt.Chart(daily)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                             color=ACCENT)
                    .encode(
                        x=alt.X("date:T", title=None),
                        y=alt.Y("cost_usd:Q", title="Вартість, $"),
                        tooltip=[alt.Tooltip("date:T", title="Дата"),
                                alt.Tooltip("cost_usd:Q", title="Вартість, $",
                                           format=".4f")],
                    )
                    .properties(height=220)
                )
                st.altair_chart(chart, width="stretch")

            by_model = (
                pd.concat([
                    view[["extract_model", "cost_usd", "input_tokens", "output_tokens"]]
                        .rename(columns={"extract_model": "Модель"}),
                    view[["group_model", "cost_usd", "input_tokens", "output_tokens"]]
                        .rename(columns={"group_model": "Модель"}),
                ])
                .groupby("Модель")
                .agg(Прогонів=("cost_usd", "count"),
                     Вартість=("cost_usd", "sum"),
                     Токени=("input_tokens", "sum"))
                .reset_index()
                .sort_values("Вартість", ascending=False)
            )
            with st.expander("📊 Розбивка по моделях", expanded=False):
                st.dataframe(by_model, hide_index=True, width="stretch",
                            column_config={
                                "Вартість": st.column_config.NumberColumn(format="$%.4f"),
                            })

            st.markdown("**🧾 Історія прогонів**")
            table = view.sort_values("timestamp", ascending=False)[[
                "timestamp", "product_line", "provider", "extract_model",
                "group_model", "reviews", "phrases", "groups", "calls",
                "cache_hits", "input_tokens", "output_tokens", "cost_usd",
                "cost_known"]].rename(columns={
                    "timestamp": "Час", "product_line": "Лінійка",
                    "provider": "Провайдер", "extract_model": "Модель (екстракція)",
                    "group_model": "Модель (групування)", "reviews": "Відгуків",
                    "phrases": "Фраз", "groups": "Груп", "calls": "Викликів LLM",
                    "cache_hits": "З кешу", "input_tokens": "Токени (вхід)",
                    "output_tokens": "Токени (вихід)", "cost_usd": "Вартість, $",
                    "cost_known": "Ціна відома",
                })
            st.dataframe(table, hide_index=True, width="stretch",
                        column_config={
                            "Вартість, $": st.column_config.NumberColumn(format="$%.4f"),
                        })

            csv = view.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Завантажити CSV", data=csv,
                               file_name="usage_history.csv", mime="text/csv")
