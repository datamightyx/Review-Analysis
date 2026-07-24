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
                               consolidate_taxonomy, merge_sibling_rows)
from pipeline.excel_writer import write_workbook
from pipeline.llm import LLM, set_max_concurrency
from pipeline.models import Taxonomy, product_key
from pipeline.precedents import (GatePrecedents, aggregate_rule_weights,
                                 load_gate_precedents, rebuild_shared_weights)
from pipeline import domain as domain_mod
from storage.db_client import (product_db, root_db, close_product_db,
                               PRODUCT_DB_NAME, ROOT_DB_NAME)
from storage import r2_sync

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


def load_phrase_counts(folder: Path) -> dict:
    """(category, product, normalized quote) -> скільки відгуків так сказали.
    Джерело — phrases.json (сирі витягнуті фрази до злиття)."""
    p = folder / "phrases.json"
    if not p.exists():
        return {}
    counts: dict = {}
    for ph in json.loads(p.read_text(encoding="utf-8")):
        key = (ph["category"], ph["product"], normalize(ph["quote"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


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
    write_workbook(tax, products, out)
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
from pipeline.similarity import set_synonym_families
set_synonym_families(cfg.get("synonym_families"))

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
                shutil.rmtree(folder, ignore_errors=True)
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

tab_prod, tab_run, tab_res, tab_by_prod, tab_review, tab_fix, tab_cost = st.tabs(
    ["📦 Продукти й PDF", "▶️ Запуск", "🗂 Результати", "🔎 Фрази товару",
     "🕵 Перевірити", "✏️ Корекції", "💰 Витрати"])


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
                use_cutoff = st.checkbox("Лише відгуки після дати")
                cutoff = st.date_input("Дата відсічення",
                                       value=date(2025, 1, 1),
                                       disabled=not use_cutoff)
            with c3:
                fresh = st.checkbox("🔄 Таксономія з нуля (--fresh)",
                                    help="Інакше нові фрази наповнюють існуючі групи")
                model = st.text_input("Модель (порожньо = з config.yaml)", value="")

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
                    llm_group = LLM(model=model.strip() or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("effort", "medium"),
                              provider=provider)
                    llm_extract = LLM(
                              model=model.strip() or cfg.get("extract_model") or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("extract_effort") or cfg.get("effort", "medium"),
                              provider=provider)
                    llm_consolidate = LLM(
                              model=model.strip() or cfg.get("consolidate_model") or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("consolidate_effort") or cfg.get("effort", "medium"),
                              provider=provider)
                    # narrow lexical dedup of new groups — its own (lower) effort
                    llm_verify = LLM(
                              model=model.strip() or cfg.get("model"),
                              cache=db,
                              effort=cfg.get("verify_effort") or cfg.get("effort", "medium"),
                              provider=provider)
                    # replay against a FINAL taxonomy (matching, not building) —
                    # its own (lower) effort; reasoning output is the main cost
                    llm_reassign = LLM(
                              model=model.strip() or cfg.get("reassign_model") or cfg.get("model"),
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
                        verify_llm=llm_verify)

                    if cfg.get("consolidate", True):
                        st.write("　консолідація таксономії…")
                        actions = consolidate_taxonomy(tax, llm_consolidate)
                        for a in actions:
                            st.write(f"　· {a}")
                        audit.extend({"type": "consolidate", "category": "",
                                      "action": a} for a in actions)

                    # 4. reassignment against the final taxonomy
                    if cfg.get("reassign", True):
                        st.write("**4/5 Перепризначення** (фінальна таксономія)")
                        rbar = st.progress(0.0, text="…")
                        r_actions = reassign_phrases(
                            phrases, tax, llm_reassign,
                            batch_size=cfg.get("group_batch", 25),
                            progress=lambda cat, d, t: rbar.progress(
                                min(d / max(t, 1), 1.0), text=f"{cat}: {d}/{t}"),
                            audit=audit)
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

                    # 5. excel
                    st.write("**5/5 Excel**")
                    out = folder / (f"{folder.name} - "
                                    f"{date.today().strftime('%d.%m.%Y')}.xlsx")
                    write_workbook(tax, products, out)

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

        fix_rn, fix_rc, fix_mg, fix_mv, fix_mc = st.tabs(
            ["✏️ Перейменувати групу", "🏷 Перейменувати фразу", "🔀 Злити групи",
             "📌 Перенести фразу", "🔗 Злити фрази"])

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

        st.divider()
        n_rules = sum(len(ov.get(k, ())) for k in
                     ("rename", "rename_canonical", "merge_groups",
                      "move_canonical", "merge_canonicals"))
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
