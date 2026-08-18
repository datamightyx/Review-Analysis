"""Wrapper page: runs the standalone 'review-scoring' Streamlit app as a
page inside this multipage app.

runpy.run_path re-executes the target script fresh on every Streamlit
rerun (required — Streamlit reruns the whole page each interaction) and
sets its __file__ correctly, so the target's own ROOT-relative paths
(products/, config.yaml, its pipeline/storage packages) keep resolving
inside "review-scoring", untouched by this repo's root files.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auth import render_logout, require_auth  # noqa: E402

# logout_button=False: цільовий app.py сам викликає st.set_page_config, а будь-яка
# st-команда до нього ламає Streamlit. Кнопку виходу малюємо після запуску.
require_auth(logout_button=False)

TARGET = ROOT / "review-scoring" / "app.py"

runpy.run_path(str(TARGET), run_name="__main__")

render_logout()
