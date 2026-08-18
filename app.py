import streamlit as st

from auth import require_auth

st.set_page_config(
    page_title="Amazon Reports",
    page_icon="📊",
    layout="wide",
)

require_auth()

st.title("📊 Amazon Reports")
st.markdown("---")

st.markdown("""
### Оберіть звіт у меню ліворуч:

- **Returns Analysis** — аналіз причин повернень по ASIN (з файлу)
- **Returns API** — те саме, але дані тягне SP-API: обираєш лише дати
- **Review Scoring** — витяг і групування фраз з відгуків (PDF) у таксономію та Excel

---
*Для переходу між звітами використовуйте бічну панель навігації.*
""")
