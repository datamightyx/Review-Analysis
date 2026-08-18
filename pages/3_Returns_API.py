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
    parse_customer_returns,
    refunds_to_return_rows,
)

st.set_page_config(page_title="Returns Analysis (API)", page_icon="🔌", layout="wide")
st.title("🔌 Returns Analysis — напряму з SP-API")
st.markdown("---")

missing_env = [k for k in REQUIRED_ENV if not os.environ.get(k)]
if missing_env:
    st.error(
        "Немає доступів SP-API у `.env`: **" + ", ".join(missing_env) + "**\n\n"
        "Потрібні: SP_API_LWA_CLIENT_ID, SP_API_LWA_CLIENT_SECRET, "
        "SP_API_REFRESH_TOKEN, SP_API_MARKETPLACE_ID, SP_API_ENDPOINT."
    )
    st.stop()

CFG = {k: os.environ[k] for k in REQUIRED_ENV}

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
@st.cache_data(show_spinner=False, ttl=3600)
def load_returns(start_s, end_s):
    return fetch_customer_returns(CFG, start_s, end_s)


@st.cache_data(show_spinner=False, ttl=3600)
def load_refunds(start_s, end_s, lag_days):
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
        raw = load_returns(start_s, end_s)
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
            ref = load_refunds(start_s, end_s, lag)
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
