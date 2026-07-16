"""Wrapper page: runs the standalone 'review-scoring' Streamlit app as a
page inside this multipage app.

runpy.run_path re-executes the target script fresh on every Streamlit
rerun (required — Streamlit reruns the whole page each interaction) and
sets its __file__ correctly, so the target's own ROOT-relative paths
(products/, config.yaml, its pipeline/storage packages) keep resolving
inside "review-scoring", untouched by this repo's root files.
"""
import runpy
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "review-scoring" / "app.py"

runpy.run_path(str(TARGET), run_name="__main__")
