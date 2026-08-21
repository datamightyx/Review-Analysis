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
import sys
import os

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import require_auth  # noqa: E402
from services import (  # noqa: E402
    create_service,
    ReturnsAnalysisError,
    ValidationError,
    AuthenticationError,
    APIError,
    RateLimitError,
    ReportError,
    DataError,
)

st.set_page_config(page_title="Returns Analysis (API)", page_icon="🔌", layout="wide")
require_auth()

# Custom CSS for credential badges
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

st.title("🔌 Returns Analysis — напряму з SP-API")
st.markdown("---")

# Initialize service
service = create_service()

# ── 0. Credentials ──────────────────────────────────────────────────────────────
CFG, CFG_FP = service.render_credentials()

# Check for missing credentials
missing_fields = service.credentials_manager.validate_complete(CFG)
if missing_fields:
    st.error(
        "Заповни доступи SP-API у панелі ліворуч: **" + ", ".join(missing_fields) + "**"
    )
    st.info(
        "Або постійним джерелом — `.env` поруч з `app.py`:\n\n"
        "```\nSP_API_LWA_CLIENT_ID=...\nSP_API_LWA_CLIENT_SECRET=...\n"
        "SP_API_REFRESH_TOKEN=...\nSP_API_MARKETPLACE_ID=ATVPDKIKX0DER\n"
        "SP_API_ENDPOINT=https://sellingpartnerapi-na.amazon.com\n```\n\n"
        "На Streamlit Cloud `.env` не існує — там те саме через Settings → Secrets:\n\n"
        "```toml\n[sp_api]\nlwa_client_id = \"...\"\nlwa_client_secret = \"...\"\n"
        "refresh_token = \"...\"\nmarketplace_id = \"ATVPDKIKX0DER\"\n"
        "endpoint = \"https://sellingpartnerapi-na.amazon.com\"\n```"
    )
    st.stop()

# ── 1. Date Selection ───────────────────────────────────────────────────────────
start, end, with_refunds, lag = service.render_date_selection()

# Validate inputs
try:
    service.validate_inputs(start, end)
except ValidationError as e:
    st.error(str(e))
    st.stop()

# Warning for recent end date with refunds
today = dt.date.today()
if with_refunds and (today - end).days < lag:
    st.warning(
        f"Кінець періоду ({end}) ближче ніж {lag} днів до сьогодні — частина "
        "refund-ів ще не релізнута на боці Amazon і в API їх поки немає. "
        "Цифра по свіжих днях буде занижена; перезапусти через тиждень."
    )

# ── 2. Run Analysis ─────────────────────────────────────────────────────────────
if not st.button("▶ Завантажий й проаналізувати", type="primary"):
    st.info(
        "Звіт повернень готується на боці Amazon 1–3 хв, refund-и тягнуться "
        "посторінково (~2 с на сторінку). Результат кешується на годину."
    )
    st.stop()

# Progress tracking
progress_container = st.container()
status_placeholder = st.empty()

def update_progress(message: str, percent: int):
    status_placeholder.write(f"{message} ({percent}%)")

# Run analysis with error handling
try:
    with st.status("Виконання аналізу...", expanded=True) as status:
        result = service.run_analysis(
            cfg=CFG,
            start=start,
            end=end,
            with_refunds=with_refunds,
            release_lag_days=lag,
            progress_callback=update_progress,
            cfg_fp=CFG_FP,
        )
        status.update(label="Аналіз завершено!", state="complete")

except AuthenticationError as e:
    st.error(f"🔐 Помилка автентифікації: {e}")
    st.info("Перевірте LWA Client ID, Client Secret та Refresh Token.")
    st.stop()

except RateLimitError as e:
    st.error(f"⏱️ Перевищено ліміт запитів: {e}")
    st.info("Спробуйте через кілька хвилин. Finances API має ліміт 0.5 запитів/сек.")
    st.stop()

except ReportError as e:
    st.error(f"📊 Помилка звіту: {e}")
    if e.details.get("report_id"):
        st.caption(f"Report ID: {e.details['report_id']}")
    st.stop()

except APIError as e:
    st.error(f"🌐 Помилка API: {e}")
    if e.status_code:
        st.caption(f"HTTP {e.status_code}")
    st.stop()

except DataError as e:
    st.error(f"📄 Помилка даних: {e}")
    st.stop()

except ReturnsAnalysisError as e:
    st.error(f"❌ Помилка: {e}")
    st.stop()

except Exception as e:
    st.error(f"💥 Неочікувана помилка: {type(e).__name__}: {e}")
    st.stop()

# ── 3. Display Results ──────────────────────────────────────────────────────────
service.render_results(result, start, end)