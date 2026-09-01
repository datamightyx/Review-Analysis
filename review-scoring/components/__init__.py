"""Bidirectional Streamlit custom components served as plain static files.

No npm/bundler step: `declare_component(path=...)` serves the folder as-is and
the HTML talks the Streamlit component postMessage protocol by hand (see
taxonomy_board/index.html). That keeps the component deployable on Streamlit
Community Cloud with zero extra dependencies.
"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_DIR = Path(__file__).parent

_board = components.declare_component(
    "taxonomy_board", path=str(_DIR / "taxonomy_board"))


def taxonomy_board(data: dict, height: int = 640, key: str | None = None):
    """Drag & drop USP board.

    `data` — {"groups": [...], "products": [...], "inspect": "", "product": "",
    "has_bucket": bool, "buckets": [...]} (built by board_payload in app.py;
    `has_bucket`/`buckets` drive the usage-band chip on subbucket categories).
    Returns the last operation the user performed as a dict carrying a
    "nonce" field, or None. Streamlit REPLAYS the last component value on
    every rerun, so the caller must de-duplicate by nonce.
    """
    return _board(data=data, height=height, key=key, default=None)
