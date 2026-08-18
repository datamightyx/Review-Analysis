"""Returns Analysis, повністю через SP-API — користувач обирає лише дати.

Два джерела, обидва з API:
  * GET_FBA_FULFILLMENT_CUSTOMER_RETURNS_DATA — фізичні повернення з коментарями
  * /finances/v0/financialEvents (RefundEventList) — refund-и (повернені гроші)

Report type 1202 (GET_DATE_RANGE_FINANCIAL_TRANSACTION_DATA) для цього акаунта
недоступний ("Request for report type 1202 is not allowed at this time"), тому
refund-и беруться з Finances API, а не зі звіту транзакцій.

Обидва вікна — UTC, тож межі діапазону збігаються (у Unified Transaction CSV
час у PT, через що краї доби можуть відрізнятись на кілька рядків).
"""

import datetime as dt
import hashlib
import io
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from returns_analysis_api import (  # noqa: E402
    REQUIRED_ENV,
    build_workbook,
    fetch_customer_returns,
    fetch_refund_events,
    get_access_token,
    parse_customer_returns,
    refunds_to_return_rows,
)

st.set_page_config(page_title="Returns Analysis (API)", page_icon="🔌", layout="wide")
st.title("🔌 Returns Analysis — напряму з SP-API")
st.markdown("---")

st.markdown(
    """
    <style>
      .cred-badge {display:inline-block; padding:2px 10px; border-radius:999px;
                   font-size:0.78rem; font-weight:600; letter-spacing:.02em;}
      .cred-ok      {background:#DCF3E3; color:#12633A;}
      .cred-partial {background:#FDF0D0; color:#8A5B00;}
      .cred-missing {background:#FBDDDD; color:#8B1D1D;}
      .cred-source  {color:#6B7280; font-size:0.78rem; margin-top:-6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 0. Доступи SP-API ──────────────────────────────────────────────────────────
#
# Значення з .env — лише початкове заповнення. Введене в панелі живе в
# st.session_state (пам'ять сервера, на диск не пишеться) і перекриває .env для
# цієї сесії. Секрети не попадають у ключ кешу: туди йде лише короткий
# відбиток (fingerprint), див. cache_key нижче.

ENDPOINTS = {
    "North America (NA)": "https://sellingpartnerapi-na.amazon.com",
    "Europe (EU)":        "https://sellingpartnerapi-eu.amazon.com",
    "Far East (FE)":      "https://sellingpartnerapi-fe.amazon.com",
}

MARKETPLACES = {
    "ATVPDKIKX0DER":  "US — amazon.com",
    "A2EUQ1WTGCTBG2": "CA — amazon.ca",
    "A1AM78C64UM0Y8": "MX — amazon.com.mx",
    "A2Q3Y263D00KWC": "BR — amazon.com.br",
    "A1F83G8C2ARO7P": "UK — amazon.co.uk",
    "A1PA6795UKMFR9": "DE — amazon.de",
    "A13V1IB3VIYZZH": "FR — amazon.fr",
    "APJ6JRA9NG5V4":  "IT — amazon.it",
    "A1RKKUPIHCS9HS": "ES — amazon.es",
    "A1805IZSGTT6HS": "NL — amazon.nl",
    "A21TJRUUN4KGV":  "IN — amazon.in",
    "A1VC38T7YXB528": "JP — amazon.co.jp",
    "A39IBJ37TRP1C6": "AU — amazon.com.au",
}
CUSTOM_MP = "Інший (ввести вручну)"

FIELD_LABEL = {
    "SP_API_LWA_CLIENT_ID":     "LWA Client ID",
    "SP_API_LWA_CLIENT_SECRET": "LWA Client Secret",
    "SP_API_REFRESH_TOKEN":     "Refresh Token",
    "SP_API_MARKETPLACE_ID":    "Marketplace ID",
    "SP_API_ENDPOINT":          "Endpoint",
}


def mask(value, keep=4):
    """Показати хвіст секрета, не розкриваючи його."""
    if not value:
        return "—"
    return f"{'•' * 8}{value[-keep:]}" if len(value) > keep else "•" * len(value)


def credentials_panel():
    """Сайдбар з доступами. Повертає (cfg, fingerprint, sources)."""
    env_cfg = {k: (os.environ.get(k) or "").strip() for k in REQUIRED_ENV}

    with st.sidebar:
        st.subheader("🔐 Доступи SP-API")

        cfg = {k: st.session_state.get(f"cred_{k}", env_cfg[k]) for k in REQUIRED_ENV}
        filled = [k for k in REQUIRED_ENV if cfg[k]]
        if len(filled) == len(REQUIRED_ENV):
            badge, text = "cred-ok", "готово до запиту"
        elif filled:
            badge, text = "cred-partial", f"заповнено {len(filled)} з {len(REQUIRED_ENV)}"
        else:
            badge, text = "cred-missing", "не заповнено"
        st.markdown(
            f'<span class="cred-badge {badge}">{text}</span>', unsafe_allow_html=True
        )

        from_env = [k for k in REQUIRED_ENV if env_cfg[k]]
        if from_env:
            st.markdown(
                f'<div class="cred-source">З <code>.env</code> підхоплено: '
                f'{len(from_env)} з {len(REQUIRED_ENV)}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Змінити доступи", expanded=not filled):
            st.text_input(
                FIELD_LABEL["SP_API_LWA_CLIENT_ID"],
                value=cfg["SP_API_LWA_CLIENT_ID"],
                key="cred_SP_API_LWA_CLIENT_ID",
                placeholder="amzn1.application-oa2-client....",
            )
            st.text_input(
                FIELD_LABEL["SP_API_LWA_CLIENT_SECRET"],
                value=cfg["SP_API_LWA_CLIENT_SECRET"],
                key="cred_SP_API_LWA_CLIENT_SECRET",
                type="password",
            )
            st.text_input(
                FIELD_LABEL["SP_API_REFRESH_TOKEN"],
                value=cfg["SP_API_REFRESH_TOKEN"],
                key="cred_SP_API_REFRESH_TOKEN",
                type="password",
                placeholder="Atzr|....",
            )

            region_names = list(ENDPOINTS)
            current_ep = cfg["SP_API_ENDPOINT"]
            ep_index = next(
                (i for i, n in enumerate(region_names) if ENDPOINTS[n] == current_ep), 0
            )
            region = st.selectbox("Регіон", region_names, index=ep_index, key="cred_region")

            mp_options = list(MARKETPLACES) + [CUSTOM_MP]
            current_mp = cfg["SP_API_MARKETPLACE_ID"]
            mp_index = mp_options.index(current_mp) if current_mp in MARKETPLACES else (
                len(mp_options) - 1 if current_mp else 0
            )
            mp_choice = st.selectbox(
                "Маркетплейс",
                mp_options,
                index=mp_index,
                format_func=lambda m: MARKETPLACES.get(m, m),
                key="cred_marketplace_choice",
            )
            if mp_choice == CUSTOM_MP:
                marketplace = st.text_input(
                    FIELD_LABEL["SP_API_MARKETPLACE_ID"],
                    value=current_mp if current_mp not in MARKETPLACES else "",
                    key="cred_marketplace_custom",
                    placeholder="ATVPDKIKX0DER",
                ).strip()
            else:
                marketplace = mp_choice

            st.caption(
                "Введене тут живе тільки в цій сесії й на диск не пишеться. "
                "Щоб зберегти назавжди — впиши у `.env` поруч з `app.py`."
            )

        cfg["SP_API_ENDPOINT"] = ENDPOINTS[region]
        cfg["SP_API_MARKETPLACE_ID"] = marketplace
        cfg = {k: (v or "").strip() for k, v in cfg.items()}

        sources = {
            k: ("панель" if cfg[k] and cfg[k] != env_cfg[k] else ".env" if env_cfg[k] else "—")
            for k in REQUIRED_ENV
        }

        st.markdown("**Поточні значення**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Поле": FIELD_LABEL[k],
                        "Значення": cfg[k] if k in
                        ("SP_API_MARKETPLACE_ID", "SP_API_ENDPOINT") else mask(cfg[k]),
                        "Джерело": sources[k],
                    }
                    for k in REQUIRED_ENV
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

        if st.button("Перевірити з'єднання", use_container_width=True):
            missing_now = [FIELD_LABEL[k] for k in REQUIRED_ENV if not cfg[k]]
            if missing_now:
                st.error("Не заповнено: " + ", ".join(missing_now))
            else:
                try:
                    token = get_access_token(cfg)
                    st.success(f"LWA токен отримано ({mask(token, 6)})")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

    fingerprint = hashlib.sha256(
        "|".join(cfg[k] for k in REQUIRED_ENV).encode("utf-8")
    ).hexdigest()[:12]
    return cfg, fingerprint


CFG, CFG_FP = credentials_panel()

missing_fields = [FIELD_LABEL[k] for k in REQUIRED_ENV if not CFG[k]]
if missing_fields:
    st.error(
        "Заповни доступи SP-API у панелі ліворуч: **" + ", ".join(missing_fields) + "**"
    )
    st.info(
        "Або поклади їх у `.env` поруч з `app.py`:\n\n"
        "```\nSP_API_LWA_CLIENT_ID=...\nSP_API_LWA_CLIENT_SECRET=...\n"
        "SP_API_REFRESH_TOKEN=...\nSP_API_MARKETPLACE_ID=ATVPDKIKX0DER\n"
        "SP_API_ENDPOINT=https://sellingpartnerapi-na.amazon.com\n```"
    )
    st.stop()

# ── 1. Вибір дат ───────────────────────────────────────────────────────────────
today = dt.date.today()
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Початок періоду", value=today - dt.timedelta(days=90), max_value=today)
with col2:
    end = st.date_input("Кінець періоду", value=today, max_value=today)

with_refunds = st.checkbox(
    "Додати refund-и без фізичного повернення (Finances API)",
    value=True,
    help="Звіт повернень містить лише те, що фізично приїхало на FC. Refund-и "
         "включають returnless refunds, товар, який покупець не відправив назад, "
         "і повернення ще в дорозі.",
)

lag = st.slider(
    "Запас на пізній реліз транзакцій, днів",
    min_value=0, max_value=14, value=7, disabled=not with_refunds,
    help="Finances API датує refund моментом РЕЛІЗУ транзакції, а не моментом "
         "повернення грошей — реліз відстає на кілька днів. Без запасу останній "
         "тиждень періоду недорахується. Зворотний бік: у вибірку потраплять "
         "refund-и, зроблені вже після кінця періоду, але релізнуті в ці дні.",
)

if start > end:
    st.error("Початок періоду пізніше за кінець.")
    st.stop()
if (end - start).days > 180:
    st.warning("Finances API віддає щонайбільше ~180 днів історії. Звузь діапазон.")

st.caption(f"Період: **{start} .. {end}** (UTC), {(end - start).days + 1} днів")

if with_refunds and (today - end).days < lag:
    st.warning(
        f"Кінець періоду ({end}) ближче ніж {lag} днів до сьогодні — частина "
        "refund-ів ще не релізнута на боці Amazon і в API їх поки немає. "
        "Цифра по свіжих днях буде занижена; перезапусти через тиждень."
    )


# ── 2. Завантаження з API (кеш по діапазону дат) ───────────────────────────────
# cfg_fp у сигнатурі — щоб зміна доступів скидала кеш; самі секрети в ключ кешу
# не потрапляють


@st.cache_data(show_spinner=False, ttl=3600)
def load_returns(start_s, end_s, cfg_fp):
    return fetch_customer_returns(CFG, start_s, end_s)


@st.cache_data(show_spinner=False, ttl=3600)
def load_refunds(start_s, end_s, lag_days, cfg_fp):
    # ~100 подій на сторінку, 0.5 запиту/с — за квартал це кілька хвилин,
    # тому показуємо лічильник сторінок
    placeholder = st.empty()
    ref = fetch_refund_events(
        CFG, start_s, end_s,
        release_lag_days=lag_days,
        progress=lambda page, rows: placeholder.write(
            f"сторінка {page}, refund-позицій: {rows}"
        ),
    )
    placeholder.empty()
    return ref


if not st.button("▶ Завантажити й проаналізувати", type="primary"):
    st.info(
        "Звіт повернень готується на боці Amazon 1–3 хв, refund-и тягнуться "
        "посторінково (~2 с на сторінку). Результат кешується на годину."
    )
    st.stop()

start_s, end_s = start.isoformat(), end.isoformat()

with st.status("Звіт повернень (SP-API Reports)...", expanded=True) as status:
    st.write("Створення й очікування звіту GET_FBA_FULFILLMENT_CUSTOMER_RETURNS_DATA")
    try:
        raw = load_returns(start_s, end_s, CFG_FP)
    except Exception as e:
        status.update(label="Помилка звіту повернень", state="error")
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

    df = parse_customer_returns(raw)
    if df.empty:
        status.update(label="Порожній звіт", state="error")
        st.error("За цей період повернень немає.")
        st.stop()
    st.write(f"Отримано {len(df)} повернень, {df['asin'].nunique()} ASIN")
    status.update(label=f"Повернення: {len(df)} рядків", state="complete")

n_refund_added, skipped_skus = 0, []
if with_refunds:
    with st.status("Refund-и (Finances API)...", expanded=True) as status:
        try:
            ref = load_refunds(start_s, end_s, lag, CFG_FP)
        except Exception as e:
            status.update(label="Помилка Finances API", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

        st.write(f"Отримано {len(ref)} refund-позицій, {int(ref['quantity'].sum())} юнітів")
        extra, skipped_skus = refunds_to_return_rows(ref, df, verbose=False)
        n_refund_added = len(extra)
        df = pd.concat([df, extra], ignore_index=True)
        status.update(label=f"Refund-и без повернення: {n_refund_added}", state="complete")

# ── 3. Аналіз ──────────────────────────────────────────────────────────────────
with st.spinner("Побудова звіту..."):
    wb = build_workbook(df)
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)

n_returns = len(df) - n_refund_added

c1, c2, c3, c4 = st.columns(4)
c1.metric("Всього кейсів", len(df))
c2.metric("Фізичні повернення", n_returns)
c3.metric("Refund без повернення", n_refund_added)
c4.metric("ASIN", df["asin"].nunique())

if n_refund_added:
    st.info(
        f"Додано **{n_refund_added}** refund-ів без фізичного повернення. "
        "Коментарів у них немає за визначенням, тому mismatch-аналіз рахується "
        "лише по фізичних поверненнях. Частина з них — повернення ще в дорозі "
        "(вікно ~30 днів), тому при повторному запуску пізніше цифра зменшиться."
    )
if skipped_skus:
    st.caption(f"SKU без жодного повернення (не змаплені на ASIN, пропущені): {skipped_skus}")

st.download_button(
    label="⬇ Завантажити результат (.xlsx)",
    data=buffer,
    file_name=f"Returns_Analysis_{start_s}_{end_s}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)
