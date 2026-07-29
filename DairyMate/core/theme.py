"""
Visual system.

Palette is drawn from the subject rather than a stock dashboard: cold parlour
steel and milk-white for the surfaces, a clinical teal for a clear result, an
oxidised rust for a flagged one, and brass for anything uncertain. Type pairs
Archivo (signage-grotesque, carries the headings) with Inter for reading and
IBM Plex Mono for every number, so figures never blend into prose.
"""
from __future__ import annotations

import streamlit as st

INK = "#0F1A24"
CANVAS = "#EEF2F4"
SURFACE = "#FFFFFF"
LINE = "#D5DEE3"
MUTED = "#5C6B76"
HEALTHY = "#0B6E5F"
MASTITIS = "#B23A26"
CAUTION = "#9A6A11"
STEEL = "#2C4356"

STATUS = {
    "healthy": HEALTHY,
    "mastitis": MASTITIS,
    "uncertain": CAUTION,
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
    color: {INK};
}}

h1, h2, h3, h4 {{
    font-family: 'Archivo', system-ui, sans-serif !important;
    letter-spacing: -0.015em;
    color: {INK};
}}

section[data-testid="stSidebar"] {{
    background: {INK};
    border-right: 1px solid {INK};
}}
section[data-testid="stSidebar"] * {{ color: #DDE6EB; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] a {{ color: #9FD8CC !important; }}

.dm-masthead {{
    border-top: 3px solid {INK};
    padding-top: 0.9rem;
    margin-bottom: 1.6rem;
}}
.dm-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: {MUTED};
}}
.dm-title {{
    font-family: 'Archivo', sans-serif;
    font-weight: 700;
    font-size: 1.95rem;
    line-height: 1.15;
    margin: 0.15rem 0 0.3rem 0;
}}
.dm-sub {{ color: {MUTED}; font-size: 0.95rem; max-width: 62ch; }}

.dm-card {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 3px;
    padding: 1.05rem 1.15rem;
}}
.dm-card--flush {{ padding: 0.85rem 1.05rem; }}

.dm-verdict {{
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    border-left: 5px solid {MUTED};
    padding: 0.55rem 0 0.55rem 0.95rem;
    margin-bottom: 0.9rem;
}}
.dm-verdict__label {{
    font-family: 'Archivo', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    line-height: 1.1;
}}
.dm-verdict__score {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    color: {MUTED};
}}

.dm-stat {{ margin-bottom: 0.85rem; }}
.dm-stat__k {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {MUTED};
}}
.dm-stat__v {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: {INK};
}}
.dm-stat__note {{ font-size: 0.78rem; color: {MUTED}; }}

.dm-note {{
    border-left: 3px solid {CAUTION};
    background: #FBF6EC;
    padding: 0.7rem 0.9rem;
    font-size: 0.86rem;
    color: {INK};
    border-radius: 0 3px 3px 0;
}}
.dm-note--info {{ border-left-color: {STEEL}; background: #EFF3F6; }}
.dm-note--ok {{ border-left-color: {HEALTHY}; background: #EDF5F3; }}

.dm-kv {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: {MUTED};
}}
.dm-kv b {{ color: {INK}; font-weight: 600; }}

.dm-caption {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {MUTED};
    margin-bottom: 0.35rem;
}}

div[data-testid="stDataFrame"] {{ font-family: 'IBM Plex Mono', monospace; }}
.stButton > button {{
    border-radius: 2px;
    font-weight: 600;
    border: 1px solid {INK};
}}
hr {{ border-color: {LINE}; }}
</style>
"""


def inject(page_title: str = "Dairy Mate") -> None:
    st.set_page_config(
        page_title=page_title,
        page_icon="◍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
